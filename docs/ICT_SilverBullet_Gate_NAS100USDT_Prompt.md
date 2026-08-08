# ICT Silver Bullet 回测实验室 → Gate NAS100USDT 部署任务书

> 这份文件是根据你上传的压缩包内容整理出来的。压缩包里是 22 张截图，内容是一套非常严谨的"分层 prompt"，用来指挥 Claude Code 搭建一个 **ICT Silver Bullet 策略**（ICT 概念：公允价值缺口/流动性扫荡/市场结构转变等）在期货（ES、NQ）上的回测实验室，设计者显然是把它当"论文级别"的统计检验在做（盲测频率调优、holdout 留存样本、Benjamini-Hochberg 多重检验校正、空模型对照、消融实验、Deflated Sharpe……），而不是"随便回测一下看着不错就上线"。我原样保留了这套方法论的严谨度（这正是判断 Silver Bullet 这类论述到底有没有真实统计边际、还是叙事巧合的关键），把散落的截图拼接、翻译、补全为连续的 Phase 0–8，然后新增了 **Phase 9**：只有通过前面的检验，才把它对接到你 Gate 交易所账户，交易 NAS100USDT。

---

## ⚠️ 先说清楚

- 我不是投资顾问，这不构成投资建议。**统计显著不等于一定赚钱**：一个策略即使完整通过下面这一整套非常苛刻的检验，也不保证未来实盘盈利——市场机制会变，Gate 的 NAS100USDT（指数永续合约）2026年1月才上线，历史数据很短，和 CME 的 NQ 期货并不是同一个东西。
- NAS100USDT 是带杠杆的衍生品，可能出现快速甚至本金全部亏损的情况。
- Phase 9 里我特意加了"不能自动跳过的人工确认关卡"——在你亲手把配置里的开关打开、并且亲手在终端打字确认之前，系统不应该下出第一笔真实资金的订单。请不要为了"更快看到实盘效果"而删掉这些关卡。
- 截图是视频/教程滚动过程中抽样截取的，其中 **Phase 4** 部分有两处小缺口（`as_taught_1m`、`as_traded` 两个命名配置的具体参数，以及验证小节里的 a)、b) 两项），我在对应位置用「⚠️」做了透明标注，不是我编的以假乱真的内容——你需要在粘贴给 Claude Code 之前，自己补一下或让 Claude Code 帮你合理补全。

---

## 如何使用这份文件

1. 确认你满足下面「先决条件」清单里的每一项。
2. 打开 Claude Code，新建一个空目录作为项目根目录。
3. **按顺序**、**一次只粘贴一个 Phase 的代码块**（每个 Phase 标题下面用围栏代码块包起来的那段英文内容，整块复制）给 Claude Code，等它完全跑完、你看过它的输出/产物之后，再粘贴下一个 Phase。不要把 Phase 1–9 一次性全部粘进去——原设计里到处都是"STOP""CHECKPOINT"这种人工复核点，一次性丢进去会让 Claude Code 自己把这些复核点也跳过去。
4. 看到 **STOP** / **CHECKPOINT** 字样时，真的停下来读它给你的结果，觉得合理再继续。
5. **Phase 9 在 Phase 8 的 holdout 结果没有通过其内部的 go/no-go 关卡之前，不要开始**——这也是你原本的要求："回测有效的话就接入"。
6. 全文用点数/最小变动单位/R 倍数计价，禁止任何地方用价格百分比——这是原设计的核心纪律之一，Phase 9 延续了同一条纪律。

---

## 关键术语速查

| 术语 | 含义 |
|---|---|
| FVG (Fair Value Gap) | 公允价值缺口，三根K线形成的价格失衡区间 |
| Liquidity sweep | 流动性扫荡：价格短暂刺穿前高/前低后又收回 |
| Displacement | 位移：相对近期波幅明显偏大的一根/一段K线 |
| MSS (Market Structure Shift) | 市场结构转变：价格突破最近的摆动高/低点 |
| Killzone | ICT定义的特定交易时段（伦敦、纽约早盘、纽约午盘） |
| R multiple | 盈亏相对于初始止损距离的倍数 |
| Sharpe ratio | 风险调整后收益指标 |
| Holdout | 整个开发过程都不能碰的"考卷"数据，只在最后跑一次 |
| Benjamini-Hochberg FDR | 多重检验的错误发现率校正，防止"测的参数组合够多，总能撞出几个好看结果" |
| Deflated Sharpe Ratio | 把"你到底测试了多少个参数组合"也计入统计显著性判断的校正方法（Bailey & López de Prado） |

---

## 先决条件（开始前必须准备好）

1. **ES 和 NQ 的 1 分钟历史数据**（CME 期货，已购买）：每个品种 3 个文件——`{ROOT}_1m_unadjusted.parquet`（真实成交价，换月时有跳空）、`{ROOT}_1m_backadjusted.parquet`（连续调整后序列）、`{ROOT}_rolls.csv`（换月记录）。如果你还没有这份数据，可以先让 Claude Code 帮你规划怎么从数据商（例如 Databento、Firstrate Data、Kibot、CME DataMine 等，请自行核实价格与授权条款）获取并整理成这个格式，再回到 Phase 1。
   - 为什么不直接用 NAS100USDT 自己的历史数据回测？因为 Gate 的 NAS100 指数永续是 2026 年 1 月才上线的新品种，历史太短，撑不起 Phase 5 那种需要成千上万笔交易、多年跨度、holdout 留存的统计检验。NQ（纳斯达克100期货）是市场公认历史最深、和 NAS100 指数相关性最高的替代品，所以**回测阶段仍然用 NQ/ES**，只在最后 Phase 9 把结论"翻译"到 Gate 的合约上。
2. Python 环境（pandas、pyarrow、numpy 等；建议 Claude Code 自己用虚拟环境管理）。
3. 一个 Gate（gate.com / gate.io）账户，并且已经生成 **仅开通合约交易权限、关闭提现权限**的 API Key/Secret（建议同时开启 IP 白名单）。这一步先做，Key 到 Phase 9 才会用到。
4. 心理预期：Phase 5 的参数扫描按设计会跑到 2 小时量级的机器时间，请预留算力和时间。

---

## 全局铁律（贯穿所有 Phase，Claude Code 应该始终记住）

- **禁止使用价格百分比或价格比率**做任何阈值/距离/结果计算——一律用点数（points）、最小变动单位（ticks）或 R 倍数。这是让"后复权(backadjusted)价格序列"能被安全使用的前提，因为加法调整不改变点数距离，但会改变比率。
- **禁止任何形式的前视偏差(look-ahead)**，唯一的例外是明确标注为"perfect bias"的对照项，且必须在所有输出里显眼地标出它是前视偏差。
- **频率/覆盖率调优必须在完全看不到盈亏结果的情况下完成**（blind to PnL）。
- **Holdout（最近 2 年）在开发阶段全程不可访问**，只在 Phase 8 手动、整体地跑一次。
- 通过统计检验的"幸存者"配置必须在**样本外品种（ES）上独立验证**，不允许因为验证结果不理想就回头重新挑选配置（no reselection）。
- 绝不能因为跑出来的结果"不好看"就偷偷改动 Phase 5 的参数网格或判定门槛。

---

## Phase 0 · 项目结构

```
I'm building a backtesting lab to test the ICT Silver Bullet strategy on futures,
and if it survives rigorous testing, extending it into a monitored live trading
system on Gate's NAS100_USDT perpetual contract. Set up this project structure:

ict_lab/
  data/raw/        my purchased 1-minute bars (ES and NQ)
  data/clean/      processed parquet output
  features/
  engine/
  configs/
  runs/
  analysis/
  live/            Gate deployment code — build this only when we reach Phase 9
  logs/

Initialize a git repo, a Python virtual environment, and a requirements file with
pandas, pyarrow, numpy, scipy, and statsmodels. Do not build anything under live/
yet. Wait for the next block.
```

---

## Phase 1 · 数据与交易时段（Data and sessions）

对应原素材 Layer 1。把你已经购买好的 ES/NQ 1 分钟数据接入，建立复权规则、UTC→美东时区转换、CME session 与 killzone 标注、数据质量报告，并且预留最近 2 年作为永不触碰的 holdout。

```
Start with the data layer.

Build a data module that does the following:

1. I have 6 files in data/raw/, three per symbol for ES and NQ:
   {ROOT}_1m_unadjusted.parquet    real traded prices, jumps at each roll
   {ROOT}_1m_backadjusted.parquet  continuous series, roll gaps removed
   {ROOT}_rolls.csv                one row per roll: date, old contract,
                                    new contract, price adjustment applied

   Both parquet files share identical timestamps, volume, and contract
   columns. They differ only in open/high/low/close.

   Build a loader that returns the BACKADJUSTED series by default and takes
   price_series="unadjusted" as an argument. Backadjusted drives all strategy
   logic. Unadjusted is used only for chart rendering and manual verification
   against a live chart.

   Assert that the timestamp index of the two parquet files matches exactly
   per symbol. Fail loudly if not.

   From rolls.csv, add an is_roll_day boolean column to both series. Keep the
   roll log accessible so I can reconcile the two price series.

   Print the min and max close per year of the backadjusted series for each
   symbol, so I can see how large the cumulative adjustment has grown in the
   early years.

2. PROJECT-WIDE RULE: no calculation anywhere may use a percentage of price
   or a price ratio. Every threshold, distance, and result is expressed in
   points, ticks, or R multiples. This is what makes the backadjusted series
   safe to use, since additive adjustment preserves point distances but not
   ratios. Write a test that scans the codebase and fails if any indicator
   or metric divides by close, open, high, or low.

3. Timestamps are UTC. Convert to US/Eastern with correct DST handling. Do
   not hardcode any UTC offset. Every downstream session calculation depends
   on this being right.

4. Label the CME session. A session runs from 18:00 ET the prior calendar
   day through 17:00 ET, with a maintenance break from 17:00 to 18:00. Assign
   every bar a session_date. Bars from 18:00 ET onward belong to the next
   session_date.

5. Add boolean columns for:
   - rth (09:30 to 16:00 ET)
   - killzone_london (03:00 to 04:00 ET)
   - killzone_ny_am (10:00 to 11:00 ET)
   - killzone_ny_pm (14:00 to 15:00 ET)

6. Data quality report. Print and save:
   - date range covered per symbol
   - count of missing minutes inside each session, by year
   - duplicate timestamps
   - bars with zero volume
   - bars where high < low, or open/close outside the high-low range
   - days with fewer bars than expected in the 10-11am ET window
   - list of any full trading days missing
   - count of roll days per year

7. Cache the cleaned output to data/clean/ as parquet, partitioned by symbol,
   series type, and year.

8. Write a verification script that prints raw OHLCV for a date and time
   window I pass in, from either series, so I can eyeball it against a chart.

9. Reserve the last 2 years as a holdout. Add a config flag for the cutoff
   date. Every function that loads data must exclude the holdout by default
   and require an explicit override to include it. I do not want to touch
   that data until the very end.

Use pandas and pyarrow. Add type hints. Write tests for session labeling and
DST edge cases, specifically spring forward and fall back days.
```

---

## Phase 2 · ICT 特征检测器（ICT feature detectors）

对应原素材 Layer 2。这一层只负责"识别"ICT概念（FVG、流动性、扫荡、位移、结构转变、高周期偏向），全部参数化，因为这些概念本来就没有唯一公认的定义。

```
Now build the feature layer. This detects the ICT concepts that make up the
Silver Bullet setup. Everything must be parameterized, because there is no
single agreed definition of any of these and I'm going to test a range.

All detectors run on the backadjusted series. All thresholds in points or
ticks, never percentages.

Build detectors for:

1. FAIR VALUE GAP
   Three consecutive bars. Bullish FVG when bar 1 high < bar 3 low. Bearish
   when bar 1 low > bar 3 high. Record gap top, gap bottom, midpoint, size in
   points, size as a multiple of ATR(14), and bar index.
   Parameters: min_size_points, min_size_atr_mult, timeframe (1m, 5m, 15m).
   Track mitigation: the bar index where price first trades back into the
   gap, and where it fully fills.

2. LIQUIDITY LEVELS
   Compute and track:
   - prior session high and low
   - prior RTH high and low
   - pre-window high and low (session open to killzone start)
   - swing highs and lows via an N-bar fractal, N configurable
   Levels remain active until swept.

3. LIQUIDITY SWEEP
   Price trades through a liquidity level and then closes back on the
   original side within K bars. Parameters: which level types count, K
   (default 3), minimum penetration in ticks.

4. DISPLACEMENT
   A bar or run of bars with range large relative to recent activity.
   Parameter: multiple of ATR(14). Test 1.0 / 1.5 / 2.0 / 3.0. Also support
   a percentile version: top X% of bar ranges over a lookback.

5. MARKET STRUCTURE SHIFT
   Price breaks the most recent swing high (bullish) or swing low (bearish)
   in the direction opposite the sweep. Configurable: close-through vs
   wick-through.

6. HTF BIAS
   Several definitions, all switchable:
   - none
   - prior day close above or below prior day open
   - 15m or 1h trend via last swing structure
   - daily 20-period MA slope
   - perfect (uses the actual direction from window start to session close;
     this is look-ahead and must be clearly flagged as such everywhere it
     appears in output)

CRITICAL: no look-ahead anywhere except the "perfect" bias. Every detector
may only use bars at or before the current index. Write a test that runs
each detector twice, once on full data and once on data truncated at bar N,
and asserts the outputs match up to bar N.

Finally, build a plotting function that renders a candlestick chart for a
given date and killzone window, using the UNADJUSTED series so prices match
what a viewer would see on their own chart. Overlay every detected feature:
FVGs as shaded boxes, liquidity levels as horizontal lines, sweeps as
markers, displacement bars highlighted, MSS as a vertical line. Generate
charts for 15 random days so I can visually confirm the detectors are
marking what a human would mark.
```

---

## Phase 3 · 执行引擎（Execution engine）

对应原素材 Layer 3。把探测到的"设置(setup)"变成真实的模拟交易：入场、止损止盈、成交规则、成本模型、交易日志与"没交易日志"。

```
The signal pipeline already exists: engine/signals.py builds the canonical
setup sequence (eligibility, bias gate, sweep, MSS, optional displacement,
first direction-matching FVG) from the FeatureStore. Now build the execution
layer that turns signal logs into trades. Do not rebuild or modify the
detectors or the precompute cache.

SCOPE: at most one trade per session per killzone window: the first valid
setup in the sequence. No re-entries.

ENTRY:
1. A signal row gives direction, FVG top/bottom/midpoint, and the timestamps
   at which everything became knowable. Entry level is a config choice:
   proximal edge of the gap, 50%, or distal edge.
2. Place a limit order at the entry level, active from the bar after the
   setup is confirmed. It fills only if price trades THROUGH the level.
   Touching it exactly does not fill.
3. If unfilled by the end of the window, cancel.

EXITS:
4. Stop options: beyond the swing that created the FVG, beyond the gap's
   distal edge, or a fixed point distance.
5. Target options: fixed R multiple (1, 2, or 3), next opposing liquidity
   level from the cached level streams, or time-based.
6. Hard exit at a configurable time (end of window or end of RTH).

FILL RULES:
- If a bar's high and low both cross the stop and the target, the STOP
  fills first. Always. Log ambiguous_bar=True on those trades so I can count
  how many results depend on this assumption.
- Stops are market orders: apply stop slippage. Entry and target fills are
  limit orders: no slippage by default.
- All fill prices snap to the tick grid.

COSTS (defaults, all configurable):
- NQ: tick 0.25, tick value $5.00. ES: tick 0.25, tick value $12.50.
- Commission $4.00 round turn per contract.
- Stop slippage 1 tick.
- Every result is computed gross and net.

TRADE LOG, one row per trade: session date, symbol, window, direction, setup
timestamp, entry timestamp, entry price, stop price, target price, exit
timestamp, exit price, exit reason, bars held, gross PnL, net PnL, R
multiple, MAE, MFE, ambiguous_bar, is_roll_day, and the full canonical
config that produced it.

NO-TRADE LOG: for every eligible session without a trade, record the FIRST
condition that failed: no sweep, bias gate, no MSS, no displacement, no FVG,
or limit unfilled. I want to know which filter is the binding constraint and
how often.

PERFORMANCE: fills iterate over killzone bars only, never the full series.
Keep the engine array-friendly, since a later layer will batch it across
many configs at once. No per-bar Python objects.

VERIFICATION:
1. Run one config (use the consensus parameters from configs/grid.json; if
   not yet defined there, take: 5m FVG, no min gap size, sweep required on
   prior-session or pre-window H/L with K=3 and 1 tick penetration,
   displacement 1.5x ATR(14), MSS not required, entry at 50% of the gap,
   stop beyond the displacement swing, target 2R, NY AM window, no bias) on
   one month of NQ. Print every trade with the surrounding bars so I can
   hand-check entries and exits by eye.
2. Extend the cache-vs-scratch equality check end to end: it must now assert
   identical TRADE logs, not just signal logs, for 3 random configs on 1
   month.
```

---

## Phase 4 · 教学版配置、交易频率与验证（Taught configs, frequency, verification）

对应原素材 Prompt 4。⚠️ **透明提示**：素材截图在这一段有两处缺口——(a) 命名配置里 `as_taught_1m` 和 `as_traded` 两项的具体参数没有被完整截到；(b) 频率验证小节里的 a)、b) 两个子项没有被截到，只截到了 c)。我在下面用「⚠️」标出了这两处，给了合理的默认处理方式，但你应该按自己的实际交易习惯调整，尤其是 `as_traded`——它本来就应该反映"你自己实际怎么交易"，没有人能替你定义。

```
Definition changes based on research into how Silver Bullet is actually
taught, then verification. No performance results exist yet; definitions get
locked before the first sweep.

1. ENGINE: MULTIPLE TRADES PER WINDOW
   Add max_trades_per_window to the execution config: 1 (current behavior)
   or unlimited with a hard safety cap of 10.
   - One open position at a time, never overlapping.
   - Signals must emit EVERY valid setup in a window under the config's
     gates, not just the first: for sweep-gated configs, every
     direction-matching FVG following any qualifying sweep/displacement
     chain in the window.
   - The engine takes setups sequentially: while a position is open, later
     setups are skipped; after an exit, the search resumes from the next
     bar.

2. ENGINE: MULTI-WINDOW CONFIGS
   The config's window field becomes a list, e.g. [ny_am] or
   [london, ny_am, ny_pm]. The engine runs each listed window independently
   within the session; daily PnL aggregates across them.

3. FEATURES: 15-MINUTE LIQUIDITY (the taught definition)
   The taught workflow marks buyside/sellside liquidity on the 15m chart
   before the session: 15m swing highs and lows plus session and overnight
   extremes. Add:
   - 15m-timeframe swing detection and its sweep scan as new cached runs
     (incremental precompute, same pattern as the penetration addition).
   - Sweep-universe presets:
     session_refs: prior-session H/L + pre-window H/L (current)
     session_refs_plus_swings: the above plus 1m N-bar swings
     bsl_ssl_15m: session refs plus 15m swing highs/lows (the taught
       definition)
     swings_only
   - Target option "next opposing liquidity" must be able to draw from the
     same preset the sweep uses.

4. NAMED CONFIGS in configs/grid.json (replacing the old consensus
   definitions):
   - "as_taught_5m", the book version:
       5m FVG, no minimum size
       bias = 15m swing-structure trend
       sweep required, universe = bsl_ssl_15m, K=3, 1 tick penetration
       displacement 1.5x ATR(14)
       MSS not required
       entry at 50% of the gap
       stop beyond the displacement swing
       target = next opposing liquidity level; if none unswept, 2R
   - "as_taught_1m" ⚠️ [not fully captured from source — reconstruct as the
     same as_taught_5m logic with the FVG/entry timeframe changed to 1m,
     everything else identical, unless you know the original intent was
     different]:
       1m FVG, no minimum size
       bias = 15m swing-structure trend
       sweep required, universe = bsl_ssl_15m, K=3, 1 tick penetration
       displacement 1.5x ATR(14)
       MSS not required
       entry at 50% of the gap
       stop beyond the displacement swing
       target = next opposing liquidity level; if none unswept, 2R
   - "as_traded" ⚠️ [not captured from source at all — this is meant to be
     MY OWN actual discretionary variant, not a generic default. Before
     running Phase 5, replace this with parameters that reflect how I
     personally would trade Silver Bullet, still respecting the
     points/ticks/R-only rule and the no-look-ahead rule. Until I specify
     otherwise, default to as_taught_5m with MSS required as the one
     difference, and flag every output derived from this config as
     "as_traded (default placeholder, not yet user-specified)" so it is
     never confused with a verified taught definition]

5. VERIFICATION (frequency and coverage, still blind to PnL)
   a) and b): design any additional blind, PnL-free coverage/consistency
      checks you think are useful here (for example: per-symbol raw setup
      counts before window/trade-cap restrictions, or how the frequency
      diagnostic in c) changes across a couple of different
      min_size_atr_mult values) — keep them blind to performance and show
      me what you chose to check and why.
   c) FULL-SPAN FREQUENCY DIAGNOSTIC, all three named configs, NQ, entire
      non-holdout span:
      - percent of windows with at least one qualified setup, by window and
        by year
      - percent of trading days with at least one trade
      - trades per year, and for as_traded the trades-per-day distribution
      - gate/no-trade reason mix by year and by window
      COUNTS AND REASONS ONLY. No PnL, win rate, or R statistics of any
      kind. Frequency tuning stays blind to performance.
      STOP here and show me the diagnostic before anything else runs.

CHECKPOINT: bring the diagnostic to chat. The number that matters is percent
of days with at least one trade. Taught expectation: the as_taught configs
should produce a setup on most days across the three windows, but not every
window every day, since even the teaching sources say some windows yield
nothing. If day coverage lands well under half, the definitions are still
too tight and we loosen them, still blind to PnL, before the sweep.
```

---

## Phase 5 · 参数扫描、空模型与统计检验（The sweep, nulls, and statistics）

对应原素材 Prompt 5，是整套体系里分量最重的一层：严格的运行时间预算、"NQ 发现 / ES 验证"的样本外设计、参数网格规范化、大规模扫描、漏斗式筛选、Benjamini-Hochberg 多重检验校正、三种空模型对照、消融实验、"自由裁量溢价"分析，以及跨年代/换月日/滑点敏感性分析。

```
Now the testing harness. Confirm all engine and feature tests pass before
starting.

HARD RUNTIME BUDGET
2 hours wall-clock for this entire layer: 90 minutes for the sweep, 30 for
everything else. If a projection exceeds the budget, optimize code or apply
the sizing rule in Part 0. Never silently drop a stage, and never change the
grid in response to results.

DESIGN: DISCOVERY ON NQ, VALIDATION ON ES
The full sampled sweep runs on NQ only. ES runs only: the survivors, the
three named configs, the ablation ladder, and the reference nulls. Survivor
results on ES are reported raw as independent validation, no reselection.

PART 0: CANONICALIZATION, BENCHMARK, SIZING
1. Enumeration reads configs/grid.json, including the axes: sweep universe
   preset, window list, max_trades_per_window. Exclude bias=perfect from the
   sweep entirely; it exists only in the ablation ladder, labeled as
   lookahead.
2. Canonicalize each config: null out dead parameters (all sweep axes when
   sweep_required=off, liquidity-target parameters when target=fixed_R, and
   any axis with no effect given the others), then dedup on the canonical
   hash. Report raw vs canonical counts.
3. Benchmark: 50 random canonical configs through the FULL pipeline
   (signals, fills, PnL, logging). Report median seconds per config and
   projected wall-clock at the available worker count. STOP for my go-ahead
   before launching.
4. Sizing rule, applied before any results exist:
   N = min(25000, floor(90min * workers / median_sec_per_config))
   canonical configs, uniform random, one draw, fixed seed. The three named
   configs are always included on top of the sample.
   - If N < 10000, do not run. Optimize first: batch configs sharing
     run-key detector streams through a single pass over the sessions,
     vectorizing threshold filters, sequencing, and fills across the batch.
     Target N >= 10000 inside 90 minutes, re-benchmark, then run.
   The sampled set is the test population for the FDR correction.

PART 1: THE SWEEP (NQ)
- Process pool across configs; FeatureStore is read-only. Workers write
  shard parquet; a merger dedups by config hash.
- Checkpointing: on restart, skip config hashes already present.
- One row per config: config hash, all canonical parameters, trade count,
  trades per year, percent of days with a trade, win rate, average R, total
  R, gross and net PnL, gross and net Sharpe, max drawdown in R and dollars,
  profit factor, percent ambiguous_bar trades, percent roll-day trades,
  first and last trade dates.
- Sharpe convention, project-wide: annualized from the daily net PnL series
  with no-trade days as zeros. State it in the output header.
- Progress line every 500 configs with ETA.

PART 2: THE FUNNEL (NQ population)
Applied in order, counts reported at each stage:
1. Population: all sampled canonical configs.
2. Minimum 100 trades. Below that, mark insufficient_sample; the row stays
   but cannot survive.
3. Net PnL > 0.
4. Net Sharpe >= 0.5.
5. Benjamini-Hochberg FDR at 10% across the full NQ population. Per-config
   p-value: one-sided test of mean trade R > 0 via stationary block
   bootstrap on the trade R sequence, 1000 resamples, vectorized. If it
   cannot finish in 15 minutes, fall back to a one-sided t-test and say so
   in the output.
Survivors = configs passing all five stages.

ES VALIDATION
Run every survivor plus the three named configs on ES, cold. Cross-symbol
survival means: passed the NQ funnel AND stayed net profitable with net
Sharpe >= 0.5 on ES. Report both counts.

PART 3: NULL TESTS
Reference set: as_taught_5m, as_taught_1m, as_traded, the best realistic NQ
survivor, and one median-Sharpe NQ config.
a) RANDOM ENTRY, SAME WINDOWS: coin-flip direction at a uniform random
   minute inside the config's window(s), stop distance and target structure
   copied from the matched config's trades, same trades-per-day count. 1000
   iterations, vectorized. All reference configs.
b) OTHER HOURS, SAME LOGIC: enumerate every non-overlapping 60-minute window
   in the session (about 21), run the single-window version of the config's
   logic on each, treat that set as the distribution. For multi-window
   configs, this means their one-window restriction.
c) SHUFFLED DIRECTION: keep the config's actual signal log, flip each
   trade's direction with p=0.5, re-simulate fills only. 1000 iterations.
   All reference configs.
Repeat (a) and (b) for as_taught_5m and as_traded on ES.
Report the real result as a percentile of each null distribution. If the
real result sits inside a null distribution, state that plainly.

PART 4: ABLATION LADDER
as_taught_5m parameters throughout, BOTH symbols, one table:
1. FVG entry only: full session, no sweep, no displacement, no bias
2. + sweep required (bsl_ssl_15m universe)
3. + displacement required
4. + window restriction (the three Silver Bullet hours)
5. + 15m bias (= the full as_taught_5m)
6. + perfect bias instead (LOOKAHEAD, labeled)
Columns: trades, win rate, avg R, net Sharpe, max drawdown.

PART 5: DISCRETION PREMIUM
For as_taught_5m, as_traded, and the best realistic survivor, from the trade
log:
- result taking every setup
- result with hindsight-perfect skipping of all losers
- the minimum fraction of losing trades a trader must correctly skip IN
  ADVANCE to (a) break even net, (b) reach 1.0 net Sharpe
Single clear numbers with one-line captions.

PART 6: STATISTICS AND SLICES
- Deflated Sharpe Ratio (Bailey and Lopez de Prado) for the best realistic
  config, using the count of canonical configs tested, cross-config Sharpe
  variance, and trade-level skew and kurtosis.
- Per-year table of average R and trade count for every survivor.
- Era split: 2010-2021 vs 2022-to-holdout-cutoff, for survivors and all
  three named configs.
- Roll-day sensitivity: survivor results with and without roll-day trades.
- Slippage sensitivity, analytic from trade logs, no re-simulation: stop
  slippage at 0/1/2/3 ticks AND time-exit slippage at 0/1/2 ticks.

PART 7: SUMMARY PACK
Write to analysis/summary/, each file small enough to paste into a chat:
results_configs.csv, funnel_counts.json, named_configs_report.md (all
three, both symbols, including win rate next to the commonly claimed 70-80%
band), es_validation.csv, ablation_ladder.csv, null_summary.md,
discretion_premium.txt, era_split.csv, frequency_report.csv (trades/year
and day-coverage percentiles across the population plus the named configs),
timing_report.txt.

HOLDOUT
Still off limits. Nothing in this layer loads it. I run it once, manually,
at the end.
```

---

## Phase 6 · 结果看板（Results dashboard）

对应原素材"Results dashboard"（第6个prompt）。生成一个单页HTML看板，把上面所有统计结果可视化。

```
Build a single-page HTML dashboard summarizing the study. Dark theme, clean,
readable at 1080p. NQ is the headline; ES is the validation story.

Panels:
1. The funnel: configs tested, profitable gross, profitable net, Sharpe >=
   0.5, survived FDR, survived ES validation
2. Distribution of net Sharpe across all NQ configs with the null
   distribution overlaid
3. Trade-frequency spectrum: trades/year across the population, log x-axis,
   with the three named configs marked
4. The ablation ladder, six labeled bars
5. Real result vs the three null distributions, real result marked
6. NQ vs ES net Sharpe scatter for survivors
7. Per-year heatmap of average R, survivors by year
8. Era split: 2010-2021 vs 2022 onward
9. The discretion premium, one large number, one-line caption
10. Win rate of the named configs against the taught 70-80% claim
11. Cost sensitivity: net Sharpe vs slippage assumptions
12. Percent of trades that hinged on the ambiguous-bar assumption
13. Equity curves: best realistic config with the null band shaded, plus
    as_taught_5m and as_traded on the same panel

Export every panel as a standalone 1920x1080 PNG.
```

---

## Phase 7 · 汇报环节（Reporting pass）

对应原素材"After the sweep / Reporting pass"。

```
Bring back to chat: results_configs.csv, funnel_counts.json,
named_configs_report.md, null_summary.md, ablation_ladder.csv,
es_validation.csv, discretion_premium.txt, frequency_report.csv. Not the
raw bars or full trade logs.
```

---

## Phase 8 · Holdout，只运行一次（The holdout, run once）

对应原素材"Final step"。这是唯一一次接触最近 2 年数据的机会——跑完之后不管结果好坏都不能回头改参数重跑。

```
When the story is clear, flip the holdout flag and run the top three
configs plus all three named configs on the last 2 years. One run.
Whatever it says is the final answer for this study — do not re-run with
different parameters after seeing it, and do not go back and change Part 0's
grid or Part 2's thresholds in light of it. Report the holdout numbers next
to their discovery/validation counterparts, plainly, including if they
disagree.
```

---

## Phase 9 · 从回测到 Gate NAS100USDT 部署（新增，仅在 Phase 8 通过后开始）

这一层是我新增的，原素材里没有——它把"论文级别的回测"和你真正想要的"接入 Gate 交易所交易 NAS100USDT"连起来，但中间不是简单地"复制策略去下单"，而是：**先设一道客观的及格线，再做"NQ点数→NAS100USDT自己的tick"换算适配，再模拟盘跑一段时间，最后才允许真实下单，且真实下单前必须有你本人手动确认**。

几个必须知道的现实差异（我已经写进下面的 prompt 里了，不是我随口一提）：
- Gate 的 NAS100USDT 是**指数永续合约**，2026年1月才上线，**7×24小时交易**，没有 CME 那种交易时段/换月，但有**资金费率(funding rate)**成本，且**杠杆上限只有 1–10 倍**（远低于一般crypto合约动辄100倍）。
- 因为上线时间短，NAS100USDT 自己的历史数据不足以再做一次 Phase 5 那种大规模统计检验——所以策略的"统计有效性"证据仍然主要来自 NQ/ES 回测，Gate 上的数据只做"频率/行为是否合理"的一致性检查，这一点必须在系统里如实标注，不能假装它和 NQ 回测一样可信。
- tick size、合约乘数、最小下单量、当前杠杆档位这些都应该在运行时向 Gate API 实时查询，不要硬编码——这些参数本来就会变。

```
I now want to take whatever survives the holdout in the previous phase and,
ONLY if it clears an explicit bar, extend it into a monitored paper-trading
and then live-trading system on Gate's NAS100_USDT perpetual contract. Do
not build any live order-placement code until the go/no-go gate below is
evaluated and passes.

PART A: GO/NO-GO GATE (evaluate before writing any live-trading code)
Using the holdout results from the previous phase (top three configs plus
three named configs, last 2 years, single run):
1. A config only qualifies for deployment work if ALL of the following hold
   on the holdout:
   - net PnL > 0
   - net Sharpe >= 0.5 (same bar as the in-sample funnel)
   - net Sharpe on holdout is not less than 50% of its net Sharpe on the NQ
     discovery population (flag and report the ratio; this checks for
     regime break / overfitting rather than picking a new threshold after
     the fact)
   - trade count on holdout is at least 20 (otherwise mark
     insufficient_sample_holdout and do not qualify)
2. Print a one-page go/no-go report: which configs qualify, which don't and
   why, with the exact numbers above. If zero configs qualify, STOP here,
   tell me plainly that nothing cleared the bar, and do not write any code
   for Parts B-F below.
3. If one or more configs qualify, list them ranked by holdout net Sharpe
   and wait for me to pick which single config to carry into deployment. Do
   not auto-select the top one for me.

PART B: INSTRUMENT ADAPTER FOR GATE NAS100_USDT
The chosen config's thresholds were calibrated in NQ points/ticks/ATR.
Before anything trades on Gate, build an adapter, not a re-fit:
1. Pull live contract specs from Gate's REST API (GET
   /futures/usdt/contracts/NAS100_USDT — official Python SDK `gate-api`,
   pip install gate-api, base URL https://api.gateio.ws/api/v4): tick size,
   contract/order size multiplier, minimum and maximum order size, current
   leverage bracket limits, and — if the field exists on this contract —
   funding interval and current funding rate. Never hardcode any of these;
   refresh at process start and periodically.
2. Write an explicit conversion layer: NQ-points to NAS100_USDT price-ticks,
   using each instrument's own tick size, not a fixed ratio assumed once.
   Convert stop distance, target distance, and displacement/ATR thresholds
   through this layer. Log both the original NQ-calibrated numbers and the
   converted NAS100_USDT numbers on every signal.
3. NAS100_USDT trades 24/7 with no CME-style maintenance break and no
   contract rolls. Simplify the session model accordingly: keep the
   killzone clock times (they are NY/London time-of-day concepts, not
   CME-session concepts) but drop is_roll_day and the 17:00-18:00 ET
   maintenance-break logic for this instrument.
4. Add funding cost to the net-PnL calculation for every open position that
   crosses a funding timestamp, using the rate actually recorded at that
   timestamp from the API, not an assumed average.
5. Replace the NQ/ES cost model (tick value, $4 commission, 1-tick
   slippage) with Gate's real numbers for this contract: pull the account's
   current maker/taker fee tier from the API rather than hardcoding a rate,
   and estimate realistic slippage from the recent order book / trade tape
   for NAS100_USDT rather than reusing the CME assumption.
6. Because NAS100_USDT only began trading around January 2026, there is not
   enough native history for another full statistical study like the sweep
   phase. Do not attempt one. Instead, pull all available NAS100_USDT
   history via GET /futures/usdt/candlesticks and run only a consistency
   check: does the chosen config produce signals at a similar frequency
   (setups per day/window) and similar average R to what the NQ backtest
   implies, within a wide tolerance? Report the comparison plainly,
   including if the sample is too small to conclude anything — that is an
   acceptable answer here, and must be reported as such rather than glossed
   over.

PART C: PAPER-TRADING ENGINE (no real orders)
1. First check whether Gate's own Demo Trading (virtual-balance)
   environment currently supports NAS100_USDT. If yes, prefer it — it
   exercises the real order-matching and API surface.
2. If not, build a local paper-trading engine: consume Gate's real-time
   market data (WebSocket ticker/order book/candles for NAS100_USDT), run
   the exact same signal and fill logic as the earlier execution engine
   (reused, not reimplemented), simulate fills against the real live order
   book, but never call any order-placement endpoint.
3. Log every simulated trade in the same TRADE LOG schema as before, tagged
   mode=paper, plus the go/no-go metrics from Part A recomputed on the
   paper sample as it accumulates.
4. Minimum trial before Part D unlocks: both of (a) at least 4 calendar
   weeks running continuously, and (b) at least 20 simulated trades.
   Whichever takes longer. Do not shorten this to move faster.
5. At the end of the trial, print a paper-vs-backtest divergence report:
   trade frequency, win rate, average R, and net Sharpe, paper vs. what the
   holdout and Part B's consistency check implied. Flag clearly if paper
   results are outside a wide, pre-stated tolerance band — don't quietly
   proceed either way.

PART D: LIVE EXECUTION LAYER (build the code, but see the hard gate in Part
E before it may run)
1. Build the Gate order-execution client using the official gate-api Python
   SDK. Read the API key and secret only from environment variables (e.g.
   GATE_API_KEY, GATE_API_SECRET); never write them to a config file, a
   log, or a committed file. Assume the key is scoped to futures trading
   only, with withdrawal permission disabled, and document that requirement
   in a README.
2. Translate the entry/exit/fill rules from the execution engine onto
   Gate's real order types as closely as Gate's API allows: limit entry
   order at the configured level; stop-loss and take-profit as the
   exchange's native conditional/stop orders where available, otherwise
   software-managed with a watchdog loop; a hard time-based exit that
   force-closes at market if still open at the configured cutoff.
3. Every order carries a client-generated idempotent order ID; on any retry
   after a network error, check order status by that ID before resending,
   so a timeout can never produce a duplicate position.
4. On every process start (including after a crash), reconcile local state
   against Gate's actual open orders and positions before doing anything
   else; if they disagree, halt and alert rather than guessing.
5. Position sizing: risk a fixed fraction of current account equity per
   trade (default 0.25%, configurable, hard-capped at 1% regardless of
   config), converted to a contract size using the live tick/contract-size
   data from Part B. Never resize a position to force a fill within a
   leverage limit — if the implied size would need more than a configured
   max leverage (default 3x, hard-capped by Gate's own ceiling on this
   contract) to fit the account's margin, reduce size or skip the trade; do
   not increase leverage to compensate.
6. Hard risk limits, checked before every order: a max-daily-loss figure
   (default 2% of equity) and a max-consecutive-loss count (default 4) that
   both disable new entries (existing positions still manage to their
   stop/target) until I manually re-enable trading.
7. Log every order request, every fill, every rejection, and every error,
   with timestamps, to a file that is never truncated automatically. Send
   an alert (write the alerting mechanism as a pluggable interface —
   webhook, email, or a simple local notification — and let me choose which
   one to wire up) on every error, every halt, and every trade.

PART E: THE HARD GATE BEFORE REAL MONEY
1. Add a config flag LIVE_TRADING_ENABLED that defaults to false and is
   never set to true by any code path automatically — only by me editing
   the config file myself.
2. Even with that flag true, the very first order placement in a fresh run
   must block on an interactive confirmation that requires me to type the
   exact contract name and the word CONFIRM; log the response either way.
3. Until both of those are true, the live runner must run in a mode that
   logs the order it would have placed and does not call any
   order-placement endpoint. Show me a week of that log before I decide
   whether to flip the flag.

PART F: STAGED CAPITAL AND A SEPARATE KILL SWITCH
1. Staged rollout, each stage requiring my explicit go-ahead to advance,
   not an automatic timer: (a) Gate demo or local paper trading (Part C),
   (b) the exchange's own minimum contract size with real funds, for at
   least 20 trades or 4 weeks, whichever is longer, (c) a size I specify,
   re-evaluated monthly against the same divergence report as Part C step
   5.
2. If live results at any stage fall outside the tolerance band from Part
   C step 5, or the max-daily-loss / max-consecutive-loss limits from Part
   D step 6 trigger twice in a rolling 30 days, automatically drop back to
   paper trading and alert me; do not auto-resize down and keep trading
   live.
3. Build live/kill_switch.py as a completely standalone script: cancels
   every open order and closes every open position on NAS100_USDT for this
   API key, then writes LIVE_TRADING_ENABLED=false back into the config,
   then exits. It must not import the strategy/engine code, only the Gate
   client, so it still works if the main process is broken.
```

---

## 落地建议与风险提示

- **诚实地对待"不及格"的结果**。这套流程本来就是为了检验 Silver Bullet 是否有真实边际而设计的——原设计里反复出现"blind to PnL""no reselection""whatever it says is the final answer"，就是为了防止你（或 Claude Code）在看到不理想的结果后偷偷调参数、换品种、扩大容忍区间直到它"看起来能用"。如果 Phase 8 或 Phase 9 Part A 没有通过，最诚实、最有价值的结论就是"这个版本的 Silver Bullet 在统计上站不住"，这本身就是值得知道的答案，不需要勉强凑一个能上线的配置。
- 即使一切顺利通过，也请按 Phase 9 设计的节奏走：模拟盘 ≥ 4 周或 ≥ 20 笔交易 → 交易所最小仓位 ≥ 4 周或 ≥ 20 笔 → 你自己指定的仓位，每月复核。不要因为等不及而跳级。
- 请确认在你所在地区、通过 Gate 进行杠杆化自动化交易是合法合规的，并且清楚 Gate 的具体条款（尤其是这类新上线不久的指数永续合约）。
- 建议先把 Phase 9 生成的 `live/` 代码请人（或另开一次对话）做一次专门的安全审查，重点看 API 密钥的存取方式、下单幂等性、以及 kill switch 是否真的能独立于主程序运行。
