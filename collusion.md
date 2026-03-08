# Section 5 Cartel-TBW 仿真实验规格文档（按最新需求重生成，Option A 已确认）

> **关键设定（已锁定）**
>
> - 出块：连续时间 Poisson（A1），全网期望间隔 (T)。
> - 统计：只统计 **canonical 主链最终接受块（D1）**。
> - tie：采用 (\gamma\in[0,1]) 的算力分流模型（C2）。
> - TBW：cartel 内 0 延迟共享私有块；对诚实方隐藏 (w^\*)，且 **私链领先不超过 1（B1 双发布）**；放弃规则为“**直到诚实连出两块才放弃**”（B-early_3）。
> - **Option A**：短窗“第 2 个需隐藏的区块时叛变”——计数的是 **叛变者自己**的“首块机会”第 2 次（不是 cartel 全体的第 2 次）。

---

## 分组 0：输入参数与实验矩阵

### 0.1 通用参数

- `T`：目标期望出块间隔（如 10 秒）
- `gamma`：平局时非 cartel 算力跟随 cartel 分支的比例
- `runs=10`：每个场景独立重复次数
- `target_blocks_long=2016`：长窗场景 canonical 终止高度
- **短窗场景参数（用于 1.2 与 2.2）**
  - `betray_on_nth_opportunity=3`：叛变者在其第 3 次“首块机会”叛变（Option A）
  - `post_betray_window_blocks=0`：叛变并解除关系后，立即停止挖矿模拟，统计此时每个矿池拥有的区块数量并记录，然后直接开始下一次模拟

- **多次叛变参数（用于“多次叛变再停止 / 叛变但不停止”）**
  - `betray_start_height=20`（canonical 高度阈值）
  - `q=0.3`（叛变概率）
  - `betray_threshold=5`（累计叛变次数阈值）

### 0.2 hashrate 输入（必须可直接改）

- **三矿池**：`p_b, p_s, p_h`，约束 `p_b+p_s+p_h=1`
  - cartel 初始成员：`{b,s}`，诚实矿池：`h`

- **四矿池**：`p_1, p_2, p_3, p_h`，约束 `p_1+p_2+p_3+p_h=1`
  - cartel 初始成员：`{1,2,3}`，诚实矿池：`h`
  - 指定叛变者：`traitor_id ∈ {1,2,3}`（可输入）

### 0.3 需要跑的场景集合（统一 4 类输出列）

为匹配你最终表格的 4 列，定义四类场景（每类输出“每矿池块占比均值±std”）：

1. `AlwaysCartel`（一直合谋，长窗 2016）
2. `BetrayBreakShort`（短窗：叛变者第 3 次机会叛变 → 解除关系 → 统计此前各矿池挖出区块数量）
3. `BetrayThenBreak`（长窗：叛变概率 q，从 height>=20 起；累计叛变≥5 才解除关系）
4. `BetrayTolerated`（长窗：叛变概率 q，但始终不解除关系）

**解除关系的动作按实验类型固定：**

- 三矿池：解除关系 = cartel **解体**（b,s 都变诚实，不再 TBW）
- 四矿池：解除关系 = **踢出叛变者**（traitor 变诚实；剩余两家继续 cartel，并用新的 (p) 重新算 (w^\*)）

**功能逻辑条目**

- [未完成] 定义 `SimConfig` / `PoolConfig` / `BetrayConfig` / `ScenarioConfig` 数据结构。
- [未完成] 输入校验：hashrate 和为 1；`gamma∈[0,1]`；`T>0`；短窗参数为正整数。
- [未完成] 实现实验矩阵生成器：对“三矿池/四矿池”各生成上述 4 个场景配置，并各自重复 10 次。

---

## 分组 1：链模型、canonical 规则与统计口径（D1）

### 1.1 区块结构

公开块 `Block` 至少包含：

- `id`, `parent_id`, `height`
- `miner_id`（b/s/h 或 1/2/3/h）
- `t_publish`（进入公共网络时间）
- `is_public=True`

私有块也可用同结构，但 `is_public=False`，不进入公共块树。

### 1.2 canonical 选择规则

- canonical tip：公开块树中 height 最大的 tip
- 同高平局：选择 `t_publish` 更早者所在分支为 canonical（确定性 tie-break，便于复现）

### 1.3 统计口径

- **长窗**：canonical 主链累计 2016 个块后停止；每矿池 share = `blocks_i / 2016`
- **短窗**：叛变并解除关系发生后，立即统计此前每矿池share = `blocks_i / height`

**功能逻辑条目**

- [未完成] 维护公开块树（`blocks_by_id`, `tips_set`）。
- [未完成] 实现 `get_canonical_tip()` 与 `reconstruct_chain(tip)`。
- [未完成] 实现 `count_blocks_on_chain(chain)`（按矿池计数）。

---

## 分组 2：事件驱动 Poisson 仿真引擎（含 (\gamma) 分流）

### 2.1 事件类型

- `MINE(pool_id, target_tip_id)`：某矿池在某 tip 上挖到块
- `RELEASE_CARTEL(deadline)`：cartel 隐藏块到期公开事件（若仍有效）

### 2.2 “挖矿过程列表”构造（核心）

每个时刻基于当前状态构造 `processes`，每个过程为 `(pool_id, tip_id, lambda_rate)`：

- 若不存在平局：每矿池只有一个过程（其全部算力指向其目标 tip）
- 若存在同高度两分叉（cartel tip 与 non-cartel tip）：
  - 对每个**非 cartel**矿池 i 拆分为两个过程：
    - `(i, cartel_tip,  gamma * p_i / T)`
    - `(i, honest_tip, (1-gamma) * p_i / T)`

  - cartel 成员仍把全部算力指向 cartel 控制器指定的 tip（私链 tip 或 race tip）

### 2.3 下一事件采样

对每个过程采样 `t + Exp(1/lambda_rate)`，加上 cartel deadline（若存在），取最小者执行。

**功能逻辑条目**

- [未完成] 定义 `MiningProcess` 结构与 `build_mining_processes()`。
- [未完成] 实现 `sample_next_event(processes, deadline)`（逐个指数采样取最小）。
- [未完成] 实现主循环 `simulate_one_run()`：执行事件、更新链、更新 cartel 状态、检查终止。

---

## 分组 3：Cartel-TBW 控制器（多成员共享、lead≤1、双发布、B-early_3）

### 3.1 (w^\*) 与共享

当 cartel 成员集合为 `members`，总算力 (p=\sum p_i)。一轮 TBW 发起时计算：

[
w^*(p)= -\frac{T}{p}\ln(2(1-p))
]

cartel 内部：一旦某成员挖到“首块 bn”，**0 延迟**共享给其他成员，所有 cartel 成员立即把算力切到私链 tip。

### 3.2 三态机（IDLE / WITHHOLD / RACE）

- `IDLE`：未隐藏块，成员在公共 canonical tip 挖
  - 挖到首块 `bn` 后：若不叛变 → 进入 `WITHHOLD`，设置 `deadline=t+ w*`

- `WITHHOLD`：隐藏 `bn`，全员在 `bn` 上挖 `bn1`（高度+1）
  - 若挖到 `bn1`（deadline 前）：**立即双发布** `bn` 与 `bn1` 到公共网络（同一 `t_publish`，先 bn 后 bn1），回 `IDLE`
  - 若公共链在 cartel 挖到 `bn1` 前先后公开了高度 `base+1` 与 `base+2`：**放弃**（丢弃 `bn`，取消 deadline），回 `IDLE`（B-early_3）
  - 若到 `deadline` 仍无 `bn1`：公开 `bn`，进入 `RACE`

- `RACE`：同高度分叉竞争；(\gamma) 分流由引擎处理
  - 任一方先出高度 `base+2` 即胜，回 `IDLE`

**功能逻辑条目**

- [未完成] 实现 `CartelController`（members/state/base_height/private_bn/deadline 等）。
- [未完成] 实现 `enter_withhold(bn)`、`on_deadline()`、`on_mine_member()`、`check_abort_condition()`。
- [未完成] 实现“双发布”的原子性（同时间戳、固定插入顺序）。
- [未完成] 处理成员变更：踢出后更新 `members`、重新计算后续轮次 (w^\*)。

---

## 分组 4：叛变机制与“解除关系”动作（按新需求重写）

### 4.1 “首块机会”的严格定义（用于叛变机会计数）

一次“首块机会”满足全部条件：

- cartel 当前 `state == IDLE`
- 叛变者挖到的块高度 `== public_canonical_tip.height + 1`

只有这种事件才：

- 计入 `opportunity_count_traitor += 1`
- 有资格触发“叛变/不叛变”的决策

**功能逻辑条目**

- [未完成] 实现 `is_traitor_opportunity(cartel_state, mined_height, public_tip_height, miner_id==traitor)`。

### 4.2 短窗叛变（用于 1.2 与 2.2）：第 3 次机会必叛变（Option A）

- `opportunity_count_traitor` 初始 0
- 每当叛变者在“首块机会”挖到块：
  - `opportunity_count_traitor += 1`
  - 若 `opportunity_count_traitor == 3`：**必叛变**
    - 叛变动作：该块 **立即公开** 给公共网络（不隐藏、不私发）
    - 立即执行“解除关系动作”（见 4.4）
    - 记录 `betray_triggered=True`，并记录 `canon_len_at_betray = canonical_len()`
    - 立即统计此时每个矿池已挖出区块

  - 若 `opportunity_count_traitor != 3`：按场景要求（通常仍遵守 cartel：进入 WITHHOLD）

> 你要求“叛变者挖到第 3 个需隐藏区块时叛变”，这里“需隐藏区块”严格对应“叛变者自己的首块机会挖到的 bn”。

**功能逻辑条目**

- [未完成] 在短窗场景中实现 `should_betray_short(opportunity_count==3)`（确定性）。
- [未完成] 叛变触发后启动统计。
- [未完成] 记录叛变触发高度、每矿池区块计数、窗口块明细（便于复核）。

### 4.3 长窗叛变（用于 1.3/2.3 等）：概率叛变 q + 高度门槛

当 canonical_height >= 20 且叛变者处于“首块机会”时：

- 以概率 `q=0.3` 叛变（立即公开首块）
- 否则遵守 cartel（进入 WITHHOLD）

叛变次数计数：`betray_count += 1`（只统计首块机会叛变）

**功能逻辑条目**

- [未完成] 实现 `should_betray_prob(height>=20, rng<q)`。
- [未完成] 维护 `betray_count`（只对机会叛变计数）。

### 4.4 “解除关系”动作（固定规则）

- **三矿池（b,s cartel）**：解除关系 = **解体**
  - `members = ∅`
  - 丢弃任何私有块、取消 deadline
  - 之后 b、s、h 全部诚实挖矿（不再 TBW）

- **四矿池（三 cartel）**：解除关系 = **踢出叛变者**
  - `members = {其余两家}`
  - 丢弃当前轮私有状态（建议直接回到 IDLE）
  - 叛变者之后作为诚实矿池参与公共挖矿
  - 剩余 cartel 继续 TBW，后续轮次用新的 (p) 计算 (w^\*)

**功能逻辑条目**

- [未完成] 实现 `break_cartel_dissolve()` 与 `break_cartel_kick(traitor_id)`。
- [未完成] break 发生时强制 cartel 状态清理（私有块/deadline/state 重置为 IDLE）。

---

## 分组 5：场景定义（与表格列一一对应）

### 5.1 三矿池（b,s,h）

- `AlwaysCartel`（长窗 2016）：无叛变，cartel 永久 `{b,s}`
- `BetrayBreakShort`（短窗 直到叛变发生）：指定 traitor（建议默认 `s`，但必须可配置）
  - traitor 在第 3 次机会叛变 → cartel 解体 → 统计此时已挖出区块

- `BetrayThenBreak`（长窗 2016）：traitor（默认 s）从 height>=200 起机会叛变（q=0.3），累计叛变>=5 → cartel 解体
- `BetrayTolerated`（长窗 2016）：traitor 概率叛变（q=0.3），但 cartel **始终不解体**（用于表格最后一列）

### 5.2 四矿池（1,2,3,h）

- `AlwaysCartel`（长窗 2016）：cartel `{1,2,3}`
- `BetrayBreakShort`（短窗 10）：traitor 在第 3 次机会叛变 → 踢出 traitor → 统计记录已挖出区块
- `BetrayThenBreak`（长窗 2016）：traitor 概率叛变（q=0.3），累计叛变>=5 → 踢出 traitor（剩余两家继续 TBW）
- `BetrayTolerated`（长窗 2016）：traitor 概率叛变但始终被容忍（cartel 保持三家）

**功能逻辑条目**

- [未完成] 将每个场景封装成 `ScenarioConfig`（type/mode/traitor_id/break_rule）。
- [未完成] 在引擎中按场景选择短窗或长窗终止条件。

---

## 分组 6：输出为柱状图

### 6.1 你要的“横列矿池、纵列ratio”的柱状图

对每个实验分别输出一张最终宽表（rows=ratio，cols=矿池）：

**行内容为：**

大类划分为3或4个矿池及其对应的名号，其中每个矿池有五个柱子：

1. `hashrate_p`
2. `AlwaysCartel_share_mean`（长窗，分母 2016）
3. `BetrayBreakShort_share_mean`（短窗，分母 canon_len_at_betray）
4. `BetrayThenBreak_share_mean`（长窗，分母 2016）
5. `BetrayTolerated_share_mean`（长窗，分母 2016）

### 6.2 原始数据必须保留（10 次 run 全部输出）

建议 `raw_runs.csv` 每行一个 run，至少包含：

- `experiment`（three/four）
- `scenario`（AlwaysCartel / BetrayBreakShort / BetrayThenBreak / BetrayTolerated）
- `mode`（long/short）
- `traitor_id`
- `seed`
- `gamma, T, q, betray_start_height, betray_threshold`
- `canonical_len`（long=2016；short=canon_len_at_betray）
- `blocks_pool_*`（对所有矿池展开列：例如 blocks_b, blocks_s, blocks_h）
- `share_pool_*`（对应 share）
- 调试字段：`opportunity_count_traitor`, `betray_count`, `canon_len_at_betray`

**功能逻辑条目**

- [未完成] 设计 `RunResult`（区分 long/short 的分母与窗口长度）。
- [未完成] 输出 `raw_runs.csv`（10 次原始数据）。
- [未完成] 输出 `cartel_three.png/.pdf`、`cartel_four.png/.pdf`。

---

## 分组 7：短窗窗口统计的精确定义（避免歧义）

短窗场景（1.2、2.2）统计流程必须严格如下：

1. 叛变者在其第 3 次机会叛变：该块立即公开
2. 立即执行 break（解体或踢出）
3. 记录 `canon_len_at_betray`（此刻 canonical 的长度）
4. 统计此前各矿池频数并除以canonical的长度得 share

## 分组 8: 总程序逻辑

**功能逻辑条目**

- [未完成] 实现 `get_new_canonical_blocks_since(old_tip, new_tip)`（多数情况下只新增 1 块，但需稳健）。
- [未完成] 实现 `summarize()`：long 用分母 2016；short 用分母 当前canonical长度；输出 blocks 与 share。
- [未完成] 单元测试：机会计数只在“traitor 自己的首块机会”增长（Option A）。

---

## 分组 9：样例参数（用于 sanity check）

- 三矿池：`p_b=0.4, p_s=0.26, p_h=0.34`（默认 traitor=s）
- 四矿池：`p_1=p_2=p_3=0.27, p_h=0.19`（默认 traitor=3）

**功能逻辑条目**

- [未完成] 实现 `main()`：跑两组样例、输出两张最终宽表 + raw_runs.csv。
- [未完成] 输出关键 sanity 日志：叛变是否恰好发生在叛变者第 3 次机会；程序是否自动开始运行下一个模拟试例。

---
