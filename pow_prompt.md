# TBW（受限私发延迟）策略的 PoW 链仿真实验规格文档（含代码逻辑）

> 目的：对论文 Section 4 的数学推理与关键假设做模拟验证。
> 核心：两矿池、连续时间 Poisson 出块、(\gamma=0)、只统计主链最终接受块（D1）、TBW 私链领先长度不超过 1、可选 DAA。

---

## 分组 0：实验总览与固定符号

**固定符号**

- 目标期望出块间隔：(T)（例如 10 秒），仅表示期望，不是固定周期。
- 大矿池（攻击者）算力占比：(p\in(0.5,1))，小矿池（诚实）占比 (1-p)。
- 全网出块率（连续时间）：(\lambda=\frac{1}{T\cdot D})，其中 (D) 为 difficulty（初始 (D=1)）。
- 两矿池出块率：(\lambda_A=p\lambda)，(\lambda_H=(1-p)\lambda)。

**TBW 延迟时间（每轮攻击重新计算）**

- 当前 difficulty 下期望间隔：(T\_{\text{cur}}=T\cdot D)
- 延迟：
  [
  w^*(p, D)= -\frac{T_{\text{cur}}}{p}\ln\bigl(2(1-p)\bigr)
  \quad(\text{仅在 }p>0.5\text{ 时为正})
  ]

**三组实验**

1. **无 DAA、按主链高度终止**：对每个 (p) 生成主链 2016 个块，统计大矿池主链份额（相对收益）。
2. **无 DAA、按固定时间终止**：对每个 (p) 运行到 (t=2016T)，统计大矿池主链块数（绝对收益不提升）。
3. **有 DAA、按固定时间终止**：只取 (p\in{0.65,0.7,0.75,0.8})，运行到 (t=n\cdot 2016T), (n\in{2,3,5,10})，验证长期份额提升。

**重复次数**

- 每个配置至少独立运行 10 次（不同随机种子），输出均值并保留 10 次原始数据。

**实现要求**

- 事件驱动仿真（连续时间），事件类型至少包含：`mine_A`, `mine_H`, `release_A`（A 的定时公开事件）。
- 维护“已发布区块树”与“攻击者私有区块（未发布）”。
- 维护“当前 canonical（主链）头”与 canonical 链序列（便于统计与 DAA）。

**功能清单**

- [未完成] 定义全局参数与实验网格（T、p 列表、runs=10、epoch_len=2016、n 列表）。
- [未完成] 实现随机数与可复现实验（seed 设计：按 scenario/p/run 派生）。
- [未完成] 定义统一的数据输出结构（每次 run 的 summary + 可选事件日志）。

---

## 分组 1：链模型与区块数据结构

### 1.1 区块对象（Block）

每个“已发布”区块至少包含：

- `id`：唯一标识
- `parent_id`
- `height`
- `miner`：`"A"` 或 `"H"`
- `t_publish`：进入网络时间（诚实矿池 = 挖到即发布；攻击者可能延迟发布）
- `children`（可选）：用于树结构

> 注意：仿真中的“网络观察”以 `t_publish` 为准；即便攻击者更早挖到，没发布就不影响公共链。

### 1.2 Canonical 链选择规则（C-longest + tie）

- **最长链规则**：canonical tip 为“已发布区块中高度最高者”。
- **平局（同高度多 tip）**：选择 `t_publish` 更早的 tip 所在链（对应 (\gamma=0) 下诚实矿工的“先见先挖”）。
- 一旦出现更长链，所有矿工（含诚实矿池）立即切换到更长链挖（C-longest）。

> (\gamma=0) 的含义在本仿真中体现在：平局时 canonical/tie 永远偏向先发布的那条链；由于攻击者通常后发布，平局下不占优势。

**功能清单**

- [未完成] 定义 `Block` 数据结构（建议 dataclass）。
- [未完成] 维护已发布区块索引：`blocks_by_id`、`tips_set`。
- [未完成] 实现 `get_chain_tip()`：根据（height, -t_publish）选择 canonical tip。
- [未完成] 实现 `rebuild_canonical_chain(tip)`：从 tip 反向追溯到 genesis，得到 canonical 区块 id 列表（用于统计与 DAA）。
- [未完成] 实现“tip 更新”逻辑：每次新块发布后更新 tips（父块不再是 tip）。

---

## 分组 2：矿工行为与 TBW 策略（严格对齐你的规则）

### 2.1 诚实矿池 H 的策略

- 始终在 **当前 canonical tip** 上挖矿。
- 挖到块立即发布（`t_publish = t_mine`）。

### 2.2 大矿池 A 的策略：三态机

A 具有以下状态（state machine）：

#### 状态 S0：`IDLE`（不在 TBW 中）

- A 在 canonical tip 上挖矿（与 H 同一父块）。
- 一旦 A 挖到“下一块”（高度 `tip.height+1`），进入 `WITHHOLD`：
  - 该块记为私有块 (B_n)，**不发布**。
  - 设定本轮的 release deadline：`t_deadline = t_mine + w*`（用当前 D 计算 (w^\*)）。
  - 记录本轮攻击的 `base_height = tip.height`。

#### 状态 S1：`WITHHOLD`（隐藏 (B_n)，私挖 (n+1)）

在 `WITHHOLD` 期间，A 行为：

- A 仅在私有块 (B_n) 上挖（目标高度 (n+1)），且 **私链领先不超过 1**：
  - 一旦 A 在 (B*n) 上挖到 (B*{n+1})（高度 `base_height+2`），立即执行：
    - **同时发布** (B*n) 与 (B*{n+1})，两块的 `t_publish = 当前时刻`。
    - 本轮结束，状态回到 `IDLE`。

- `WITHHOLD` 的终止条件（取最先发生者）：
  1. **成功条件**：A 先挖到 (B\_{n+1}) → 立刻发布两块（上面已述）。
  2. **失败放弃条件（你给的 B-early-3）**：诚实链先后发布高度 (n) 与 (n+1)（即 canonical 已达到 `base_height+2`），且此时 A 仍未挖到 (B\_{n+1})：
     - A **放弃本轮**：丢弃私有 (B_n)，不发布；状态回 `IDLE`，并从新的 canonical tip 开始正常挖矿，等待下一次自己先挖到首个块再发起 TBW。

  3. **到期公开条件**：到达 `t_deadline` 且 A 未挖到 (B\_{n+1})：
     - A 发布 (B_n)（`t_publish = t_deadline`），进入 `RACE`（见下）。

> 解释：你要求“遇到诚实链提前连出两块则放弃”；你也要求“若到期仍未挖到 n+1，则公开 n”。因此本仿真将两者并列为终止条件。

#### 状态 S2：`RACE`（(B_n) 已公开但与诚实块同高竞争）

- 此状态只会发生在：诚实链已经发布了高度 (n) 的块 (H_n)，且 A 在 `t_deadline` 才公开 (B_n)，导致“同高度双分叉”。
- (\gamma=0) 与 tie 规则意味着：诚实矿池 H 会继续在 (H_n) 上挖；A 会在 (B_n) 上挖。
- `RACE` 的结束：
  - 若 A 先挖到 (B\_{n+1}) 并发布，则 A 分支高度变为 `base_height+2`，canonical 切换到 A 分支，诚实分叉块（如 (H_n)）成为孤块；状态回 `IDLE`。
  - 若 H 先挖到 (H\_{n+1}) 并发布，则 canonical 固化为诚实分支，(B_n) 成为孤块；A 切换到 canonical tip，状态回 `IDLE`。

**功能清单**

- [未完成] 定义 A 的状态结构 `AttackerState`（state/base_height/private_block_ids/t_deadline 等）。
- [未完成] 实现 `start_attack_if_idle()`：A 在 IDLE 挖到块后进入 WITHHOLD 并计算 (w^\*) 与 deadline。
- [未完成] 实现 `handle_mine_A()`：按 state 分别处理（IDLE 触发 WITHHOLD；WITHHOLD 挖到 n+1 立即双发布；RACE 挖到下一块立即发布）。
- [未完成] 实现 `handle_mine_H()`：始终在 canonical tip 挖到即发布，并触发“失败放弃条件”的检测。
- [未完成] 实现 `handle_release_A()`：到 deadline 发布 (B_n)，并切换到 RACE（若此时 canonical 已到 base_height+2 则应直接放弃而不是发布）。
- [未完成] 实现 `check_abort_condition()`：若 canonical*height >= base_height+2 且 A 仍无 (B*{n+1})，立即放弃本轮（丢弃私有块、取消 release）。
- [未完成] 明确“取消 release”机制：若 state 改变（成功/放弃/进入 RACE 等），需保证不会再触发旧 deadline 事件。

---

## 分组 3：事件驱动仿真引擎（连续时间）

### 3.1 时间推进与事件选择（Poisson）

在任意时刻 (t)，给定当前 difficulty (D)：

- 全网出块率：(\lambda=1/(T\cdot D))
- A 出块率：(\lambda_A=p\lambda)
- H 出块率：(\lambda_H=(1-p)\lambda)

每一步仿真从三类“候选事件”中取最早发生者：

- A 挖到下一块：(t_A = t + \mathrm{Exp}(\lambda_A))
- H 挖到下一块：(t_H = t + \mathrm{Exp}(\lambda_H))
- A 的 release deadline（若存在）：(t_R = \text{deadline})

取 `t_next = min(t_A, t_H, t_R)`，并执行对应 handler。

> 使用每步重采样是合法的（指数分布无记忆性），可避免维护复杂的“取消事件队列”。

### 3.2 需要维护的“挖矿目标 tip”

- H 的挖矿目标 = 当前 canonical tip。
- A 的挖矿目标：
  - `IDLE`：canonical tip
  - `WITHHOLD`：私有 (B_n)
  - `RACE`：A 分支 tip（(B_n) 或其后继）

### 3.3 终止条件（按实验场景变化）

- 场景 1：canonical 链区块数达到 2016（不含 genesis）。
- 场景 2：仿真时间达到 (2016T)（到点停止）。
- 场景 3：仿真时间达到 (n\cdot 2016T)（到点停止），且期间按 epoch 做 DAA。

**功能清单**

- [未完成] 实现 `simulate(run_config) -> run_result` 主循环（按上述三事件取最早）。
- [未完成] 在主循环中集成 handler 调用与链/状态更新。
- [未完成] 实现三种终止条件的统一接口（按 `mode` 切换）。
- [未完成] 设计可选 `event_log`（用于 debug：记录每次 publish/mine/reorg/state transition）。

---

## 分组 4：Difficulty Adjustment（DAA）逻辑（仅用于实验 3）

### 4.1 Epoch 定义

- epoch 长度：2016 个 **canonical 链块**（不含 genesis）。
- 记录 epoch 起始时间 `t_epoch_start`（第 1 个 epoch 为 0 或 genesis 后的起点）。
- 当 canonical 链新增到满 2016 个块时，计算：
  - `T_total = t_publish(last_block_of_epoch) - t_epoch_start`
    （按你的选择 E-time-sum：等价于 canonical 到达该高度所用的累计时间）

### 4.2 难度更新公式（E-DAA-yes，且无 cap）

[
D_{\text{new}} = D_{\text{old}} \cdot \frac{2016\cdot T}{T_{\text{total}}}
]

- 若 TBW 使 `T_total > 2016*T`，则 (D*{\text{new}} < D*{\text{old}})，出块会变快。
- 更新后，全网出块率变为：(\lambda*{\text{new}} = 1/(T\cdot D*{\text{new}}))。

> 不使用 4× cap 或其它上下限（E-cap-no）。

### 4.3 DAA 与 reorg 的处理约束（重要）

由于你的 TBW 约束（私链领先≤1 + 放弃规则），reorg 深度应当 ≤1。仍建议实现对“最后 1 块被替换”的稳健处理：

- canonical tip 更新后应重建 canonical 链序列。
- epoch 计数应以“当前 canonical 链序列的长度”判断是否达到 2016 的倍数。
- 为避免复杂性：当 epoch 边界附近发生 reorg 时，以“canonical 链实际达到该高度时的 publish 时间”作为终点时间。

**功能清单**

- [未完成] 维护 `difficulty D` 与 `lambda` 的实时更新。
- [未完成] 实现 `maybe_adjust_difficulty(canonical_chain, t_now)`：当 canonical_len 达到 epoch 边界时计算 `T_total` 并更新 D。
- [未完成] 维护 epoch 计时：`t_epoch_start` 何时重置（每次完成 epoch 后置为当前时刻）。
- [未完成] 记录每个 epoch 的统计（T_total、D_old/D_new、epoch 内 A/H 主链块数）。

---

## 分组 5：指标统计（对应你要证明的三件事）

### 5.1 基础统计口径（所有场景都要）

统计对象：**最终 canonical 链上的块（D1）**。

每次 run 至少输出：

- `A_blocks_canonical`
- `H_blocks_canonical`
- `canonical_len`
- `A_share = A_blocks_canonical / canonical_len`
- `total_time`（run 结束时仿真时间）
- `A_orphan_published`、`H_orphan_published`（可选，用于解释为何绝对收益下降）
- `attacks_started`、`attacks_success_2blocks`、`attacks_success_race`、`attacks_abort`、`attacks_release_only`（可选但强烈建议）

### 5.2 场景 1：相对收益提升（2016 块定长）

- 对每个 (p\in{0.55,0.6,\dots,0.95})：
  - 运行到 canonical 链长度 = 2016
  - 计算 `A_share` 的 10 次均值与标准差

- 图：`x=p`，`y=mean(A_share)`，并画基线 `y=x`。

### 5.3 场景 2：绝对收益不提升（固定时间 2016T）

- 对每个 (p\in{0.55,\dots,0.95})：
  - 运行到时间 `t_end = 2016*T`
  - 得到 `A_blocks_canonical(t_end)`（此时 canonical_len 可能 ≠ 2016）

- 需要对比的基线（你选 E2）：
  [
  \text{baseline_A} = p \cdot \frac{t_{\text{end}}}{T}=p\cdot 2016
  ]
- 推荐画图形式（更“学术”且更稳）：
  - 画 **差值**：`delta = A_blocks_canonical - baseline_A`（应 ≤ 0）
  - 同时给误差棒：10 次 run 的均值 ± 标准差
    或画 **比值**：`ratio = A_blocks_canonical / baseline_A`（应 ≤ 1）

### 5.4 场景 3：有 DAA 的长期收益（固定时间 n·2016T）

- (p\in{0.65,0.7,0.75,0.8})
- 时间倍数 (n\in{2,3,5,10})，各自终止时间：
  [
  t_{\text{end}}=n\cdot 2016\cdot T
  ]
- 基线（你选 E2）：
  [
  \text{baseline_A}=p\cdot \frac{t_{\text{end}}}{T}=p\cdot n\cdot 2016
  ]
- 输出与画图建议：
  - 对每个 (p)，画 `x=n`，`y=mean(A_blocks_canonical / baseline_A)`（期望随 n 上升并可能 >1）
  - 同时输出每个 epoch 的 (D) 与 (T\_{total}) 轨迹，作为机制解释支撑

**功能清单**

- [未完成] 实现 canonical 链统计函数 `summarize_run()`（从 canonical 链序列统计 A/H 块数）。
- [未完成] 实现“孤块统计”（已发布但不在 canonical）。
- [未完成] 实现“攻击事件统计”（started/success/abort/release/race）。
- [未完成] 实现三类场景的聚合统计（10 runs 的 mean/std）。
- [未完成] 设计绘图数据表结构（方便直接喂给 matplotlib）。

---

## 分组 6：数据保存格式（必须含 10 次原始数据）

### 6.1 目录结构建议（输出到本地）

```
results/
  scenario1_no_daa_by_blocks/
    summary.csv
    raw_runs.csv
  scenario2_no_daa_by_time/
    summary.csv
    raw_runs.csv
  scenario3_daa_by_time/
    summary.csv
    raw_runs.csv
    epoch_stats.csv
figures/
  s1_relative_share.png
  s2_absolute_delta.png
  s3_longterm_ratio.png
```

### 6.2 `raw_runs.csv`（每行 = 一个 run）

字段建议（至少）：

- `scenario`
- `p`
- `run_id`
- `seed`
- `T`
- `t_end`
- `canonical_len`
- `A_blocks_canonical`
- `H_blocks_canonical`
- `A_share`
- `A_orphan_published`
- `H_orphan_published`
- `attacks_started`
- `attacks_success_2blocks`
- `attacks_success_race`
- `attacks_abort`
- `attacks_release_only`
- `final_difficulty`（场景 3）
- `num_epochs_completed`（场景 3）

### 6.3 `summary.csv`（对 raw 按 p 聚合）

- `p`
- `metric_mean`
- `metric_std`
- `runs=10`

其中 metric 随场景变化：

- 场景 1：`metric = A_share`
- 场景 2：`metric = A_blocks_canonical - p*2016`
- 场景 3：`metric = A_blocks_canonical / (p*n*2016)`（需包含 n 维度）

**功能清单**

- [未完成] 定义统一的 `RunResult` 结构并可序列化为 dict/CSV。
- [未完成] 实现 CSV 写入（raw + summary + epoch_stats）。
- [未完成] 保证每个 scenario 的输出文件名固定、可重复覆盖或带时间戳。

---

## 分组 7：绘图规范（用于论文/报告）

只规定“画什么”，不限定颜色风格（避免误导）。

### 场景 1（相对收益）

- 图 1：`p` vs `mean(A_share)`，叠加基线 `y=x`。
- 误差棒：`± std(A_share)`。

### 场景 2（绝对收益）

两种任选其一（建议差值图更直观）：

- 图 2a（差值）：`p` vs `mean(A_blocks_canonical - p*2016)`，误差棒。
- 图 2b（比值）：`p` vs `mean(A_blocks_canonical / (p*2016))`，误差棒 + 基线 `y=1`。

### 场景 3（长期收益 + DAA）

- 对每个 (p) 一条曲线：`n` vs `mean( A_blocks_canonical / (p*n*2016) )`，误差棒。
- 附图（可选）：`epoch_index` vs `difficulty D`（或 `T_total/(2016T)`），用于解释 DAA 机制驱动。

**功能清单**

- [未完成] 实现绘图函数：输入 summary 表，输出 PNG/PDF。
- [未完成] 为每张图保存“用于重现”的数据（例如同时保存 `.csv`）。

---

## 分组 8：伪代码（供 AI 直接落地实现）

```python
def simulate_one_run(T, p, mode, t_end=None, target_blocks=None, enable_daa=False, seed=None):
    rng = np.random.default_rng(seed)
    D = 1.0
    t = 0.0

    init_genesis()

    attacker = AttackerState(state="IDLE", base_height=None, private_bn=None, deadline=None, race_tip=None)

    epoch_len = 2016
    t_epoch_start = 0.0
    epochs_completed = 0

    while not termination(mode, t, canonical_len(), t_end, target_blocks):
        # current rates
        lam = 1.0 / (T * D)
        lamA = p * lam
        lamH = (1-p) * lam

        tA = t + rng.exponential(1/lamA)
        tH = t + rng.exponential(1/lamH)
        tR = attacker.deadline if attacker.has_deadline() else float("inf")

        t_next = min(tA, tH, tR)
        t = t_next

        if t_next == tR:
            handle_release_A(attacker, t)
        elif t_next == tA:
            handle_mine_A(attacker, t)
        else:
            handle_mine_H(attacker, t)

        # update canonical tip after any publication
        canonical_tip = get_canonical_tip()
        canonical_chain = rebuild_chain(canonical_tip)

        # abort condition (B-early-3): honest chain reaches base_height+2 before A finds n+1
        check_abort_condition(attacker, canonical_tip)

        # DAA
        if enable_daa:
            if len(canonical_chain) == (epochs_completed+1) * epoch_len:
                T_total = t - t_epoch_start
                D = D * (epoch_len * T) / T_total
                t_epoch_start = t
                epochs_completed += 1

    return summarize_run(...)
```

**功能清单**

- [未完成] 将上述伪代码拆分为可测试模块（state/chain/engine/metrics）。
- [未完成] 为关键函数写单元测试（尤其是：状态转换、放弃条件、canonical 选择、DAA 更新点）。

---

## 最后校验清单（避免“模拟与推导不一致”）

- 你要验证的 (w^\*) 推导假设是“Poisson 出块 + 两矿池 + (\gamma=0) + lead≤1 + 到期公开 + 若诚实连出两块则放弃”。代码必须逐条实现。
- 场景 2/3 的基线是 (p\cdot (t\_{\text{end}}/T))（你选 E2），不是 (p\cdot 2016\cdot T)（维度错误）。
- canonical 的 tie 规则必须固定，否则结果会漂移（这里用 `t_publish` 更早者胜）。

**功能清单**

- [未完成] 在代码入口打印“实验配置摘要”（p、T、w\* 计算、gamma、DAA 公式），写入日志文件。
- [未完成] 对少量样例（如 p=0.6）输出事件 log，人工检查 1~2 轮 TBW 的状态机是否符合描述。

---

如果你接下来要把 (\gamma>0) 加入敏感性分析，只需要在“平局时 canonical/tie 选择”改为：诚实矿工以概率 (\gamma) 转向攻击者分支，并相应改变 H 的挖矿目标；其余框架不变。
