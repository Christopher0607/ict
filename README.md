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
- [~] Phase 3 — execution engine: code-complete, 163 tests passing (up
      from 94). Phase 3's own text assumes two pieces of infrastructure
      already exist ("engine/signals.py builds the canonical setup
      sequence... from the FeatureStore") that neither Phase 1 nor Phase 2
      actually asked for, so this phase built them too, scoped to what's
      needed now rather than Phase 5's heavier version:
      - `engine/feature_store.py` — Phase 2's detectors, memoized per
        (detector, parameters) so a multi-config run doesn't redo shared
        work. In-memory only; Phase 5's own persistent/sharded cache is a
        separate, later concern.
      - `engine/signals.py` — the canonical setup sequence: eligibility →
        bias gate → sweep → MSS → optional displacement → first
        direction-matching FVG, each stage narrowing the candidate
        direction and advancing a reference point the next stage must
        search strictly after. At most one setup per session+window
        (Phase 3's scope; Phase 4 is what allows more).
      - `engine/execution.py` — entry (proximal/50%/distal, tick-snapped
        limit order, strict through-trade fill, cancel at window end),
        exits (swing/gap-distal/fixed-points stop; fixed-R/next-opposing-
        liquidity/time-based target; hard exit at window or RTH end),
        same-bar stop-always-wins with `ambiguous_bar` flagged, stop
        slippage, MAE/MFE, full trade log and no-trade log per the spec's
        schema.
      - `configs/strategy_config.py` — every knob as a validated,
        immutable `StrategyConfig`, plus `CONSENSUS_CONFIG`: the exact
        baseline Phase 3's VERIFICATION section names, since
        `configs/grid.json` doesn't exist yet.
      - The spec's own required check — "run every config on full data and
        on data truncated at a cutoff bar, assert they agree" — extended
        to signals *and now trade logs* in `tests/test_cache_vs_scratch.py`,
        comparing one FeatureStore reused across 3 differently-parameterized
        configs against a fresh store per config. Passed cleanly, meaning
        the cache keys are complete (no cross-config contamination).
      - `engine/run_verification.py` — runs the consensus config on one
        month of a symbol and prints every trade with its surrounding
        bars for hand-checking, per Phase 3's VERIFICATION item 1. Ready
        to run for real; still blocked on the Phase 1 data the same way
        everything else has been.
- [~] Phase 4 — taught configs, frequency, verification: code-complete,
      212 tests passing (up from 163). The frequency diagnostic's actual
      CHECKPOINT is blocked on real data the same way Phase 3's
      verification was — see "Phase 4's CHECKPOINT is not yet met" below.
      - **Multiple trades per window** (item 1): `engine/signals.py` now
        emits *every* direction-matching FVG after a qualifying sweep/
        displacement chain, uncapped — capping moved to
        `engine/execution.py`, since only execution knows which setups
        actually fill. `simulate_trades` groups signals by
        `(session_date, window)`, walks them in `setup_at` order tracking
        a "position open until" timestamp, and skips (as its own logged
        no-trade row — `position_open` or `max_trades_reached`) any setup
        that lands at-or-before the open position's exit or once
        `config.max_trades_per_window` is reached. An entry that never
        fills does not block later setups: the clock only advances on an
        actual fill.
      - **Multi-window configs** (item 2): `StrategyConfig.window` (str) is
        now `windows` (tuple) — `generate_signals` runs each window
        independently within a session; daily PnL is just the sum across
        whatever each window produced.
      - **15-minute liquidity + sweep-universe presets** (item 3):
        `features/liquidity.py`'s `swing_levels` takes a `timeframe`
        (1m or 15m, resampled and remapped through the resampled bars' own
        `knowable_at`); `configs/sweep_universe.py` adds the 4 named
        presets (`session_refs`, `session_refs_plus_swings`, `bsl_ssl_15m`,
        `swings_only`), resolved per-window since pre-window level types
        are window-specific. Two correctness gaps found while wiring this
        up and fixed before building on top of it: (a) the `next_liquidity`
        target was drawing from *every* level type regardless of which
        sweep universe was active — `_target_price` now takes the same
        resolved `target_level_types` the sweep gate used, per the spec's
        explicit "must be able to draw from the same preset the sweep
        uses"; (b) `StrategyConfig.swing_15m_n` was defined but never
        actually reached `all_liquidity_levels` — `FeatureStore`'s
        `liquidity_levels`/`sweeps` silently used a hardcoded default
        regardless of what a config specified. Both are now covered by
        dedicated tests (a level type outside the active universe is
        proven excluded from targeting; two different `swing_15m_n` values
        are proven to produce different cached results, not a silent
        collision).
      - **Named configs in `configs/grid.json`** (item 4): `as_taught_5m`
        (the book version), `as_taught_1m` (identical, 1m FVG/entry
        timeframe), and `as_traded` — replacing Phase 3's single
        `CONSENSUS_CONFIG` as the reference set going forward.
        `CONSENSUS_CONFIG` itself is untouched (Phase 3's own verification
        and tests still pin its exact values); `run_verification.py`'s
        default now points at `as_taught_5m`. Two source gaps the spec
        flagged with ⚠️ (`as_taught_1m`'s and `as_traded`'s exact
        parameters weren't fully captured from the original material) are
        resolved the way the spec itself instructs: `as_taught_1m` = the
        same `as_taught_5m` logic with FVG/entry timeframe changed to 1m;
        `as_traded` = `as_taught_5m` with MSS required as the one
        difference, defaulted and labeled everywhere as **"as_traded
        (default placeholder, not yet user-specified)"**
        (`configs/grid.py`'s `placeholder_note`) until replaced with real
        discretionary parameters. `engine/pipeline.py` factors the
        "resolve sweep universe → compute levels/sweeps → run signals/
        trades" sequence that `run_verification.py`, the frequency
        diagnostic, and the cache-vs-scratch test all need into one place,
        after a stub/key mismatch caused by the `swing_15m_n` fix showed
        three near-duplicate copies of this logic drifting out of sync.
      - **Frequency diagnostic** (item 5, `engine/frequency_diagnostic.py`):
        computes, for all three named configs — pct of windows with ≥1
        qualified setup by window/year, pct of trading days with ≥1 trade
        by year (the CHECKPOINT's "number that matters"), trades/year,
        `as_traded`'s trades-per-day distribution, and gate/no-trade reason
        mix by year/window (item c). Every function reads only
        session/window/timestamp columns — `compute_frequency_diagnostic`
        drops `gross_pnl`/`net_pnl`/`r_multiple`/`mae_points`/`mfe_points`
        from the trade log immediately after running the pipeline, before
        any metric function sees it, so "blind to PnL" is structural, not
        just a matter of what gets printed. Items (a) and (b) were left to
        us to design ("any additional blind, PnL-free coverage/consistency
        checks you think are useful here") — we implemented the spec's own
        two parenthetical suggestions directly: (a) `raw_setup_counts`,
        the total *count* of uncapped signals by window/year (not just
        window/day presence), read next to the coverage percentage to tell
        apart "this window rarely qualifies" from "it qualifies often but
        the cap/sequencing discards most of it"; (b) `atr_mult_sensitivity`,
        rerunning the day-coverage number at a couple of alternative
        `fvg_min_size_atr_mult` values against the named configs' current
        0.0 (no minimum).

  **Phase 4's CHECKPOINT is not yet met.** Item (c) is explicit: "FULL-SPAN
  FREQUENCY DIAGNOSTIC... entire non-holdout span" of real NQ data, and the
  CHECKPOINT itself asks whether real day-coverage looks like the teaching
  material. `ict_lab/data/raw/` still has no purchased data (see "Getting
  the Phase 1 data" above) — the same block that stopped Phase 3's
  verification script from running for real. The diagnostic code above is
  complete and tested against synthetic fixtures, and `run_and_print`
  prints a loud disclaimer whenever it isn't handed real data, but a
  synthetic random walk has no genuine session structure or liquidity
  behavior to measure — running it now would produce syntactically valid
  numbers that are not evidence about the real question. Phase 5 is
  similarly gated on real data for anything beyond code-completeness, so
  this project is holding at the Phase 4 CHECKPOINT until the purchase
  happens.
- [~] Phase 5 — parameter sweep, nulls, statistics: code-complete, 366
      tests passing (up from 279 before this phase). This is the heaviest
      layer in the spec, and its own CHECKPOINT is unreachable here for
      two independent reasons — see "Phase 5 cannot actually run in this
      environment" below. Every part was still built fully and tested
      against synthetic fixtures, the same discipline as every prior
      phase's data-gated work.
      - **Part 0** (`configs/sweep_grid.py`, `engine/sweep_benchmark.py`):
        `configs/grid.json` gained a `sweep_axes` section (window subsets,
        `bias_method` excluding `perfect`, `sweep_required`/`universe`,
        `mss_required`, `displacement_required`, `fvg_timeframe`,
        `entry_level`, `stop_type` restricted to swing/gap_distal,
        `target_type` restricted to fixed_r/next_liquidity, and
        `max_trades_per_window` ∈ {1, 10}) and a `sweep_fixed` section for
        the dial parameters deliberately left unswept — the spec names
        only three axes explicitly; the rest are documented, scoped
        choices, not silent assumptions. Enumeration → canonicalization
        (nulls dead parameters given a config's own state, e.g.
        `sweep_universe` when `sweep_required=False`) → dedup on a
        content hash (excluding `name`) takes the real grid from 64,512
        raw combinations to 48,384 valid to **36,288 canonical**. The
        benchmark samples 50 canonical configs through the full pipeline
        (fresh `FeatureStore` per config — the conservative, no-
        cross-config-caching number) and the sizing rule
        (`N = min(25000, floor(90min·workers / median_sec))`) decides
        whether Part 1 can even run — this is the spec's explicit
        **STOP for my go-ahead before launching**, and nothing in
        `engine/sweep_orchestrator.py` bypasses it: launching Part 1
        needs a second invocation with `--confirm-launch`, a human flag,
        never inferred.
      - **Part 1** (`engine/sweep_runner.py`): a `ProcessPoolExecutor`
        runs one config's full pipeline per task; completed rows flush to
        shard parquet every 500 configs (checkpointing — a restart skips
        hashes already in a shard) with a progress/ETA line at the same
        cadence. The one-row-per-config summary schema is exactly the
        spec's list (trade count, trades/year, day coverage, win rate,
        avg/total R, gross/net PnL and Sharpe, max drawdown in R and
        dollars, profit factor, ambiguous-bar/roll-day percentages,
        first/last trade dates), plus a `config_hash`/params. Net Sharpe
        is annualized from the daily net-PnL series with no-trade days as
        zero, stated once here as the project-wide convention. A real,
        reproducible bug surfaced while building this: **parquet
        round-trips Python tuples as numpy arrays and `None` as `NaN`**,
        which would have silently corrupted any config rebuilt from a
        shard (needed for Part 2's ES reruns) — `windows`/
        `sweep_level_types` are now JSON-encoded and `sweep_universe`'s
        `None` is written as `""`, with `config_from_row` undoing both
        and normalizing numpy scalars back to native types so a
        round-tripped config's canonical hash matches the original's
        exactly (tested, not just assumed). A lightweight companion
        r_multiples shard (config_hash → JSON trade-R-sequence) rides
        alongside the summary shard, captured for free during the same
        pipeline run, since Part 2's FDR stage needs every config's raw
        trade sequence, not just its aggregate stats.
      - **Part 2** (`engine/funnel.py`): the five stages in order — min
        100 trades, net PnL > 0, net Sharpe ≥ 0.5, then Benjamini-Hochberg
        FDR at 10% computed across the **full sampled population**, not
        just the pre-filtered survivors of the earlier stages (correcting
        only against a cherry-picked shortlist would understate how many
        hypotheses were actually tried and defeat the point of FDR
        control). Per-config p-values come from a vectorized-across-
        resamples stationary block bootstrap (1000 resamples, Politis &
        Romano) on the trade-R sequence; a timed sample projects the
        full-population wall-clock cost first, and if that projection
        exceeds the 15-minute budget the *entire* population falls back
        to a one-sided t-test instead — one consistent method throughout,
        since mixing methods within a single correction would make the
        p-values incomparable, with the method actually used reported
        alongside the result. ES validation reruns every survivor plus
        the three named configs cold (a fresh, ES-only `FeatureStore`,
        never sharing state with anything NQ-derived).
      - **Part 3** (`engine/null_models.py`): three null generators for
        the reference set (as_taught_5m/1m, as_traded, best realistic
        survivor, one median-Sharpe population config) — random-entry
        (coin-flip direction at a uniform random minute in the same
        window/day, real trade's stop/target point-distances copied and
        re-simulated via execution.py's own exit logic), other-hours
        (the config's real, unchanged logic re-run one hour at a time
        across ~20 non-overlapping 60-minute windows excluding the
        config's own killzones), and shuffled-direction (real trades,
        each direction flipped p=0.5, fills only re-simulated). The
        "other hours" test needed `_window_mask` (sessions.py) fixed for
        midnight-wraparound windows (e.g. 23:00-00:00), a real gap that
        happened to never matter until a window needed to cross
        midnight — backward-compatible, since none of the standard
        killzones wrap. Every null test shares one `FeatureStore` across
        its ~20 hour variants (detector caches key on parameters, never
        window boundaries), cutting that stage from ~13s to ~2s per
        reference config in testing.
      - **Part 4** (`engine/ablation.py`): the six-rung ladder (FVG-only →
        +sweep → +displacement → +window restriction → +15m bias
        [= the real as_taught_5m, verified by hash equality in a
        dedicated test] → +perfect bias, labeled LOOKAHEAD), holding
        every as_taught_5m parameter fixed except the four the spec
        names, so the table isolates exactly those four gates. Building
        this hit a **real, previously-latent bug in Phase 3's
        `execution.py`**: `next_liquidity` targeting's "already swept"
        filter used `DataFrame.apply(..., axis=1)`, which on a zero-row
        frame silently returns a misaligned, non-boolean empty Series
        instead of an empty boolean mask — collapsing the candidates
        frame to zero *columns*, not just zero rows, and crashing a few
        lines later. No prior phase's tests happened to hit the empty-
        candidates case; rung 1's `full_session` window (also newly
        added, permanently, to `sessions.WINDOWS`) did. Fixed with a
        vectorized `MultiIndex.isin` check and a regression test.
      - **Part 5** (`engine/discretion_premium.py`): every-setup vs.
        hindsight-perfect-skip-all-losers vs. the minimum fraction of
        losers a trader would need to skip (worst-loss-first, the
        provably optimal order) to reach breakeven net PnL or 1.0 net
        Sharpe — a plain search over that ordering, since Sharpe depends
        on the whole daily distribution, not a running total.
      - **Part 6** (`engine/statistics_slices.py`): Deflated Sharpe Ratio
        (Bailey & López de Prado) computed **entirely at the trade level**
        for internal consistency — observed Sharpe, cross-config Sharpe
        spread, and skew/kurtosis all from each config's own r_multiple
        sequence, since the spec's "trade-level skew and kurtosis" would
        otherwise sit awkwardly against a daily-annualized Sharpe on a
        different scale; kurtosis is raw (Pearson, normal=3), matching
        the paper's Gaussian asymptotic-variance formula (checked against
        it directly). Per-year tables, the 2010-2021 vs.
        2022-to-holdout-cutoff era split, roll-day sensitivity, and
        analytic (no re-simulation) slippage sensitivity at 0/1/2/3 stop
        ticks and 0/1/2 time-exit ticks — the latter by algebraically
        shifting only the affected trades' exit price by the tick delta
        and re-deriving PnL/R, not touching anything else.
      - **Part 7** (`engine/summary_pack.py`, `engine/sweep_orchestrator.py`):
        writes all ten spec-named files to `analysis/summary/`.
        `results_configs.csv` is Part 1's full per-config table by
        design (the spec names it as the sweep's raw output) and isn't
        itself "small enough to paste into a chat" at full sweep size —
        that qualifier reads as applying to the other nine genuinely
        summary-shaped files. The orchestrator CLI wires Parts 0-7 in
        order, never silently drops a stage, never touches
        `ict_lab/data/holdout.py` (checked directly, not just asserted),
        and never mutates the grid after seeing a result. A full,
        synthetic-data dry run of the entire Parts 1-7 pipeline (tiny
        population, 5 null-test iterations) is exercised end-to-end as an
        integration test, since a wiring mistake anywhere in a pipeline
        this size is far more likely to surface from actually running it
        than from unit-testing each piece in isolation — and it's exactly
        this test that caught both bugs described above.

  **Phase 5 cannot actually run in this environment, for two independent
  reasons.** First, the familiar one: `ict_lab/data/raw/` still has no
  purchased data, the same block as every prior phase's real-data work.
  Second, and new here: even setting data aside, **this sandbox's own
  throughput fails the spec's sizing rule.** A real benchmark run in this
  container measured a median 2.6s/config against 4 CPUs, giving
  `N = floor(90·60·4 / 2.6) ≈ 8,300` — below the spec's own
  `N ≥ 10,000` minimum, meaning the sizing rule *itself*, exactly as
  built, says **do not run** and calls for optimizing the pipeline first
  rather than proceeding. That's not a data problem; it's this
  environment's compute. The code took that answer at face value instead
  of working around it. Phases 6 onward remain similarly gated until both
  the data and adequate compute exist.
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
