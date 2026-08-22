# propfirm-lab · 项目行为约定

期货 propfirm 账号数学：把规则写成可执行代码，算清「买一个账号到底值多少钱」。
Python + numpy。**目标是全自动执行，但自动化政策必须先书面确认。**

## 🔴 最重要的事实

**这个项目的核心结论是：均值是正的，但中位数是负的，而且差 5 倍。**

在扣成本后期望值 = 0（即「刚好抹平手续费和滑点」）的假设下，Apex 50k：

| 指标 | 数值 |
|---|---|
| P(通过考核) | 31.9% |
| **P(这个账号最终赚钱)** | **8.3%** |
| 平均净收益 | +$134/账号 |
| **中位数净收益** | **−$131（就是手续费）** |

**两个 break-even 差 5 倍：**

- 让**均值**转正：**−0.036R**（Apex）/ −0.107R（TopStep）/ −0.090R（MFFU）
- 让**中位账号**转正、即 P(赚钱) > 50%：**+0.201R ～ +0.230R**

> 「持续长期稳定盈利」这个目标里的**「稳定」两个字，要求的是后面那个数**。
> 前面那个数只能让你在买几百个账号之后期望为正——那是彩票，不是稳定。

复现：`python run_edge_curve.py`，产出 `findings/01_edge_requirement.md`。

## 硬性约束（不要越过）

1. **不往「稳赚」「必胜」方向包装。** propfirm 的商业模式决定了总体对玩家是负 EV。
   本项目的价值在于算清临界点在哪，不在于证明它能赚。
2. **报均值必须同时报中位数和 P(赚钱)。** 这类分布右偏极严重，
   只报均值等于撒谎。任何结论都要带置信区间（`mean_net_ci`）。
3. **不许为了好看改门槛。** 判定门槛写死在 `findings/01`，之后不许调。
   发现自己在「再试一组参数看看」，停下。
4. **实盘下单必须有人工闸门。** 代码不得自行翻转 `LIVE_TRADING_ENABLED`。
   `live/` 在 Phase 0 政策确认 + Phase 4/5 通过之前不许建。
5. **负面结果是合格交付。** 「你的 edge 不够，不该买账号」是完整结论，
   写进 findings 就结束，不要试图挽救它。

## ⚠️ 自动化政策风险（未解决，Phase 0 阻塞项）

2026 年各家政策互相矛盾且在变：

- **Apex**：来源打架。有的说允许半自动/DCA 但**禁止入场出场都无人干预的全自动 bot**，
  且禁止 HFT；有的说全自动 bot 直接封号并没收资金。
- **TopStep**：自动化最友好，**有官方 API**（TopstepX，基于 ProjectX Gateway），
  文档 `gateway.docs.projectx.com`，`POST /api/Order/place`，JWT + SignalR。
  仍限制 HFT 与延迟套利。
- **MFFU**：2025-07 才解除自动化禁令。

**TopStep / ProjectX 是唯一有官方文档背书的合规全自动路径。**
在禁止全自动的平台上跑 bot 会被封号+没收，那和项目目标直接对立。
**书面确认（工单截图）存进 `findings/00_automation_policy.md` 之前，不写实盘代码。**

## 规则数据的可信度

`rules/ruleset.py` 里的数字是 2026-08 从公开页面整理的，**每条都带 `effective_date`**。
Apex 2026-03 刚整体改版过，规则变动频繁。

**以你自己账户后台的条款为准。** 用真钱之前，逐条对照后台核对一遍
`rules/ruleset.py`，不要相信这个文件。

## 工作方式要求

**写完代码必须实际运行验证。** 这个项目已经靠「真的跑一遍」抓到两个真 bug：

1. 提款时机建模错误——原版只在路径末尾检查提款资格，等于让账号复利一整年
   再一次性全额提走，把 +0.4R 的净收益虚报成 $61k（真值 $15k）。
2. 月费从来没被计算——`study_lifecycle` 没传 `months_of_fees`，
   TopStep/MFFU 的订阅费全免了。

两个都不是代码审查发现的，是跑出数字觉得不对劲、去拆中间量才发现的。
**看到好得不像话的结果，先假设是自己错了。**

## 验证纪律

`tests/test_sim_validation.py::test_gamblers_ruin_closed_form` 是硬门槛：
零 edge + static 回撤，模拟器必须复现赌徒破产解析解 `DD/(DD+target)` = 0.4545。
**这个测试挂了不许往下走**，说明模拟器的吸收边界逻辑坏了。

其余测试覆盖：三种回撤规则的严重性排序、intraday 地板对未实现峰值的棘轮效应、
trailing lock、日损上限、以及每条提款闸门的「刚好触发/刚好不触发」双向边界。

## 常用命令

```bash
pip install -r requirements.txt
python -m pytest tests -q              # 32 个测试
python run_edge_curve.py --quick       # 粗跑，约 1 分钟
python run_edge_curve.py               # 正式跑
```

## 当前状态

- ✅ Phase 1 `rules/` — Apex/TopStep/MFFU 规则可执行编码
- ✅ Phase 2 `sim/` — 路径级账号模拟器，解析解验证通过
- ✅ Phase 3 `paths/` + edge 需求曲线
- ⬜ Phase 4 `research/` — 需要行情数据（Databento $125 免费额度 / TopstepX retrieveBars）
- ⬜ Phase 5 `portfolio/` — 多账号组合
- ⬜ Phase 6 `live/` — 阻塞在自动化政策确认

**下一步不是写策略，是拿你爆掉的账号成交记录回放进模拟器**，
确认它复现真实死亡时点与死因。那是这个项目最值钱的输入。
