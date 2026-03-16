# pow-python-2026

这是一个 PoW 链仿真项目，包含两类核心实验：

1. 双矿工 TBW 攻击仿真
2. 多矿池 cartel 合谋与背叛仿真
3. 单攻击者 selfish mining 仿真

## 项目结构

### `pow_simulation.py`

双矿工仿真入口。

- 参与者只有两类：攻击者 `A` 和诚实矿工 `H`
- 攻击者采用 TBW 策略
- 支持 3 个场景：
  - `scenario1_no_daa_by_blocks`
  - `scenario2_no_daa_by_time`
  - `scenario3_daa_by_time`

### `pow_collusion.py`

多矿池 cartel 仿真入口。

- 支持三矿池和四矿池
- 支持 4 个场景：
  - `AlwaysCartel`
  - `BetrayBreakShort`
  - `BetrayThenBreak`
  - `BetrayTolerated`

### `pow_three_tolerate.py`

`pow_collusion.py` 的精简运行器。

- 只跑三矿池
- 只跑 `BetrayTolerated`
- 支持多进程并行
- 支持进度条显示

### `pow_selfish.py`

单个 selfish miner 对其余 honest miners 的经典 Eyal-Sirer 自私挖矿仿真。

- 支持扫描一组 `p`
- 支持 `gamma` tie-breaking 参数
- 输出终端表格、CSV 和 PNG 图

### `try.py`

独立的公式试算和绘图脚本，不参与主仿真流程。

### 说明文档

- `pow_prompt.md`：双矿工 TBW 实验规格
- `collusion.md`：cartel 合谋实验规格
- `selfish.md`：selfish mining 实验规格
- `pow_three_tolerate.md`：`pow_three_tolerate.py` 的维护说明

## 双矿工 TBW 仿真

实现文件：`pow_simulation.py`

### 模型

- 全网为连续时间 Poisson 出块
- 目标平均出块间隔为 `T`
- 攻击者算力占比为 `p`
- 诚实矿工算力占比为 `1-p`
- 可选难度调整 `DAA`

### 主要数据结构

- `Block`
  - 已发布区块
  - 字段：`id`、`parent_id`、`height`、`miner`、`t_publish`
- `PrivateBlock`
  - 攻击者尚未发布的私有块
- `AttackerState`
  - 攻击状态机，状态有 `IDLE`、`WITHHOLD`、`RACE`
- `AttackCounters`
  - 攻击成功、失败、公开首块等计数
- `EpochStat`
  - DAA 场景下每个 epoch 的统计
- `RunResult`
  - 单次运行结果

### 主链规则

- 选择高度最高的 tip 作为 canonical tip
- 如果高度相同，选择 `t_publish` 更早的链
- 再相同时用 block id 做稳定决策

### 攻击者状态机

#### `IDLE`

攻击者在主链 tip 上正常挖矿。

当攻击者先挖到下一块时：

- 先不发布
- 保存为私有块
- 进入 `WITHHOLD`
- 记录本轮基准高度
- 设置公开截止时间

截止时间公式：

`w*(p, D) = -(T * D / p) * ln(2 * (1 - p))`

#### `WITHHOLD`

攻击者已经有一个私有首块，继续在私链上挖第二块。

可能结果：

1. 先挖到第二块
   - 立即双发布
   - 本轮成功
2. 诚实链先推进两层
   - 放弃私链
   - 本轮失败
3. 到达截止时间还没挖到第二块
   - 只公开首块
   - 转入 `RACE`

#### `RACE`

公开网络出现两个同高度分支。

- 如果攻击者分支先再出一块，则攻击者获胜
- 如果诚实分支先继续推进，则攻击者失败

### 事件驱动流程

每一步采样三个候选事件时间：

- 攻击者挖到块
- 诚实矿工挖到块
- 攻击者截止时间到达

取最早事件执行，并更新链、状态和统计。

### 3 个场景

#### 场景 1：`scenario1_no_daa_by_blocks`

- 不启用 DAA
- 运行到 canonical chain 长度达到 `epoch_len`
- 统计攻击者主链份额 `A_share`

#### 场景 2：`scenario2_no_daa_by_time`

- 不启用 DAA
- 运行到固定时间 `epoch_len * T`
- 统计攻击者主链块数相对基准的增减

#### 场景 3：`scenario3_daa_by_time`

- 启用 DAA
- 运行到固定时间 `n * epoch_len * T`
- 观察长期收益比例变化

### DAA

当 canonical chain 新增满一个 `epoch_len` 后：

`D_new = D_old * (epoch_len * T) / t_total`

其中 `t_total` 是该 epoch 内 canonical chain 实际走完所花的时间。

## 多矿池 cartel 仿真

实现文件：`pow_collusion.py`

### 模型

- 多个矿池参与挖矿
- 某些矿池可组成 cartel
- cartel 内部共享私有块
- 竞争时支持 `gamma` 分流

### 主要数据结构

- `PoolConfig`
  - 矿池 id 和算力占比
- `BetrayConfig`
  - 背叛方式和参数
- `ScenarioConfig`
  - 场景规则
- `SimConfig`
  - 全局仿真参数
- `CartelController`
  - cartel 状态控制器
- `MiningProcess`
  - 某矿池以某个速率在某条分支上挖矿
- `NextEvent`
  - 下一事件
- `RunResult`
  - 单次运行结果

### cartel 状态

- `IDLE`
  - cartel 成员在公共主链上挖
- `WITHHOLD`
  - cartel 私藏首块，并继续挖第二块
- `RACE`
  - cartel 分支与 honest 分支同高竞争

### `gamma` 的作用

当两条同高分支竞争时，非 cartel 矿池的算力会被拆分为两部分：

- `gamma * rate` 挖 cartel 分支
- `(1 - gamma) * rate` 挖 honest 分支

### 背叛逻辑

traitor 只有在“首块机会”才可能背叛。首块机会要求：

- 当前 cartel 状态为 `IDLE`
- 挖到的是新一轮首块
- 挖矿者是指定的 traitor
- traitor 仍属于 cartel

背叛方式有两种：

- 短窗背叛
  - traitor 在第 N 次机会时直接背叛
- 概率背叛
  - 从某个高度开始，每次机会以概率 `q` 背叛

背叛动作是：

- 将本来应私藏的首块立即公开

### 4 个场景

#### `AlwaysCartel`

- 一直合谋
- 不背叛

#### `BetrayBreakShort`

- 在第 N 次机会背叛
- 背叛后立刻解体或踢出 traitor

#### `BetrayThenBreak`

- 达到高度阈值后按概率背叛
- 累计到阈值后再解体或踢出 traitor

#### `BetrayTolerated`

- 达到高度阈值后按概率背叛
- cartel 不解散

## `pow_three_tolerate.py`

这是 `pow_collusion.py` 的最小化运行版本。

### 特点

- 仅支持三矿池 `b,s,h`
- 仅支持 `BetrayTolerated`
- 支持多进程
- 单进程时显示单次 run 的 canonical block 进度条
- 多进程时显示整体 runs 完成进度条

### `runs` 和 `jobs`

- `runs`
  - 要执行多少次独立仿真
  - 每次 run 有不同随机种子
  - 用于做均值、方差、稳定性统计
- `jobs`
  - 最多同时开多少个工作进程
  - 只影响并行度，不改变统计意义

两者不是一回事：

- `runs` 决定“做多少个实验”
- `jobs` 决定“同时跑几个实验”

实际并行度是：

`min(runs, jobs)`

## 常用参数

### `pow_simulation.py`

- `--scenarios`
- `--T`
- `--runs`
- `--epoch-len`
- `--base-seed`
- `--results-dir`
- `--figures-dir`
- `--skip-plots`
- `--no-sample-event-log`

### `pow_collusion.py`

- `--T`
- `--gamma`
- `--runs`
- `--target-blocks-long`
- `--betray-on-nth-opportunity`
- `--betray-start-height`
- `--q`
- `--betray-threshold`
- `--three-pools`
- `--three-traitor`
- `--four-pools`
- `--four-traitor`
- `--seed-base`
- `--max-events`
- `--output-dir`
- `--skip-plots`

### `pow_three_tolerate.py`

- `--T`
- `--gamma`
- `--runs`
- `--target-blocks-long`
- `--betray-on-nth-opportunity`
- `--betray-start-height`
- `--q`
- `--betray-threshold`
- `--three-pools`
- `--three-traitor`
- `--seed-base`
- `--max-events`
- `--progress-step-percent`
- `--jobs`

## 输出结果

### 双矿工仿真输出

- `results/scenario1_no_daa_by_blocks/raw_runs.csv`
- `results/scenario1_no_daa_by_blocks/summary.csv`
- `results/scenario2_no_daa_by_time/raw_runs.csv`
- `results/scenario2_no_daa_by_time/summary.csv`
- `results/scenario3_daa_by_time/raw_runs.csv`
- `results/scenario3_daa_by_time/summary.csv`
- `results/scenario3_daa_by_time/epoch_stats.csv`

### cartel 仿真输出

- `results/collusion/three/raw_runs.csv`
- `results/collusion/three/summary.csv`
- `results/collusion/four/raw_runs.csv`
- `results/collusion/four/summary.csv`
- `results/collusion/raw_runs.csv`

## 运行示例

```bash
python pow_simulation.py --scenarios all --runs 100
python pow_collusion.py --runs 10 --target-blocks-long 20160
python pow_three_tolerate.py --runs 8 --jobs 8 --target-blocks-long 200000
```
