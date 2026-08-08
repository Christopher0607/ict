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
      cache. This environment cannot reach data vendor sites (Databento,
      Firstrate Data, Kibot, CME DataMine) — that step needs to happen
      outside this sandbox.
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
can't complete this step on your behalf (no agent session can reach these
vendor sites, and buying data needs your own payment action anyway). Options,
found via research, not independently verified against current pricing:

- **Kibot** — "All Futures Continuous Contracts 1-Minute Intraday Data": 83
  symbols back to 2009, adjusted + raw unadjusted columns in one CSV schema,
  **$520 one-time, no subscription**, free 3-month sample to check the format
  before buying. Simplest, most concrete option. kibot.com
- **Firstrate Data** — NQ continuous history back to 2008 (~15 years), with
  three adjustment types including an **"Absolute-Adjusted" (points-based)**
  series — the additive adjustment this project's own no-percentage rule
  requires, as opposed to their ratio-adjusted variant. firstratedata.com
- **Databento** — official CME data vendor (GLBX.MDP3), pay-as-you-go/metered
  historical API, likely the most "correct" source but priced per query rather
  than a flat one-time fee. databento.com
- **CME DataMine** — the exchange's own first-party historical data platform;
  more institutional/higher overhead to get set up with. cmegroup.com/datamine

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
