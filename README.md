# Event-Driven PoW Simulator

## System setup

This repository is an event-driven proof-of-work simulator for OCW/TBW,
classic selfish mining, and three-pool cartel experiments.

- Block arrivals are independent Poisson processes. With target interval `T`
  and difficulty `D`, the total rate is `1/(T*D)` and a miner with hash share
  `alpha` has rate `alpha/(T*D)`.
- Published blocks are visible immediately: propagation delay is assumed to be
  negligible. This zero-delay assumption matches the analytical models and
  isolates strategic withholding from network effects.
- The canonical chain is the longest chain. Equal-height public tips are ordered
  by earlier publication time and then by block ID; explicit mining races use
  the configured `gamma` tie-following fraction.
- The default OCW withholding window is `w=10T`. DAA experiments use
  `D_new = D_old * (epoch_len*T) / elapsed_time`, with either canonical or all
  published blocks counted at the boundary.

`pow_selfish.py` samples only the next block winner. This is the embedded event
chain of the same Poisson model and is sufficient because that baseline has no
time-dependent rule or DAA.

## Default experimental protocol

All repeats are independent and reproducible. Seeds are integers derived from
the first eight bytes of a SHA-256 hash of the base seed and complete parameter
point, so parallel execution does not change results.

| Program                          | Default parameter point                                                                                        |            Runs | Seed                                              | Horizon                                                            | Figure error bars                                     |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------- | --------------: | ------------------------------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------- |
| `pow_simulation.py`              | Scenario 3: `p=0.65,0.75`, canonical DAA, `n=1,2,3,5`; scenario 4: `p=0.65,0.75`, public/orphan-aware DAA, `n=1,3,10`; both OCW and SM | 10 per strategy and `p` | base `2026` + scenario + `p` + run ID + max round | Scenario 3 ends at `5*2016*T=100800`; scenario 4 ends at `10*2016*T=201600`; listed `n` values are checkpoints | sample standard deviation |
| `pow_chain_withhold.py`          | `p=0.30..0.95` by `0.05`; `w/T=0.5,1,10`                                                                       |             100 | base `2026` + `p` + `w/T` + run ID                | stop when canonical length is at least 2016                        | sample standard deviation                             |
| `ocw_w_sweep.py`                 | `alpha=0.65,0.75`; `w/T=0,0.25,0.5,1,2,5,10`                                                                   |              50 | base `2026` + `alpha` + `w/T` + run ID            | stop when canonical length is at least 2016                        | 95% CI: `1.96*s/sqrt(n)`                              |
| `pow_selfish.py`                 | `p=0.55..0.95` by `0.05`; `gamma=0`                                                                            |              10 | base `2026` + `p` + `gamma` + run ID              | cross 2016 finalized blocks, then settle the pending state         | standard error: `s/sqrt(n)`                           |
| `pow_collusion.py`               | four cartel/betrayal scenarios; pools `b=0.3,s=0.3,h=0.4`                                                      | 10 per scenario | base `20260224` + experiment + scenario + run ID  | reach 20160 canonical blocks; report the first 20160               | no error bars; CSV includes sample standard deviation |
| `pow_three_tolerate.py`          | `BetrayTolerated`; pools `b=0.30,s=0.50,h=0.20`                                                                |              10 | same scheme as `pow_collusion.py`                 | reach 2016 canonical blocks; report the first 2016                 | no figure                                             |
| `plot_orphan_rate_comparison.py` | reads OCW and SM summaries                                                                                     |            none | none                                              | inherited from input CSVs                                          | sample standard deviation from the input summaries    |

In `pow_simulation.py`, each `(scenario, strategy, p, run ID)` is one trajectory;
the listed `n` values are checkpoints, not independent runs. Scenario 3 and the
OCW part of scenario 4 report `A_blocks/(p*n*epoch_len)`, while scenario 4 SM
reports `A_blocks/(n*epoch_len)`. Error bars are the sample standard deviation
of that normalized metric across the 10 runs.

All default SM points have `p>0.5`, so `SelfishMiningDAASimulation` uses its
majority mode: attacker blocks are released in `epoch_len` batches, and an
unfinished private batch is counted as eventually canonical at a checkpoint or
the final time horizon. Scenario 3 adjusts difficulty from canonical-chain
boundaries, while scenario 4 uses the public count basis. For majority-mode SM
in scenario 4, that public-work counter includes all attacker and honest work
that will be published and retains each block's original mining time.

## Classic selfish-mining baseline

The baseline is the standard Eyal--Sirer lead-state strategy:

- withhold newly mined blocks;
- at lead 1, release one block when honest miners catch up and enter state `0'`;
- at lead 2, release the private chain to override the honest block;
- at lead 3 or more, release one block and retain the remaining lead;
- in state `0'`, an honest miner follows the attacker branch with probability
  `gamma`; otherwise it follows the honest branch. The default is `gamma=0`.

`pow_selfish.py` implements the standalone counting baseline. In
`pow_simulation.py`, `SelfishMiningDAASimulation` uses this lead-state logic only
for `p<=0.5`; its current default points (`p=0.65,0.75`) use the majority-mode
epoch release described above, so `gamma` does not affect those default runs.

## Abstracted factors

- Transaction-fee variation: omitted; all accepted blocks have equal value.
- Latency asymmetry: omitted; publication is instantaneous for every miner.
- Network topology: omitted; all miners share one global public view.
- Uncle/stale-block rewards: omitted; non-canonical blocks earn zero reward.
- Mempool dynamics, block-size effects, eclipses, and miner entry/exit are also
  outside the model.

## Python files

| File                                        | Purpose                                                                                                        |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `pow_simulation.py`                         | Runs fixed-time OCW/SM comparisons with canonical or public/orphan-aware DAA and writes raw runs, checkpoints, epoch statistics, summaries, and figures. |
| `pow_chain_withhold.py`                     | Runs the chained OCW strategy for several attacker shares and withholding windows.                             |
| `ocw_w_sweep.py`                            | Sweeps `w/T`, compares simulation with theory, and plots 95% confidence intervals.                             |
| `pow_selfish.py`                            | Runs the standalone classic Eyal--Sirer selfish-mining baseline without DAA.                                   |
| `pow_collusion.py`                          | Simulates three-pool cartel cooperation, betrayal, breakup, and tolerance.                                     |
| `pow_three_tolerate.py`                     | Provides a small runner for only the three-pool `BetrayTolerated` case.                                        |
| `plot_orphan_rate_comparison.py`            | Combines existing OCW and selfish-mining orphan-rate summaries in one figure.                                  |
| `try.py`                                    | Plots the scratch analytical term `p^2(1-p)/(1+p)`; it is not part of the simulations.                         |
| `tests/test_pow_simulation.py`              | Tests normalization, checkpoint timing, majority-mode epoch release, and both DAA count bases.                 |
| `tests/test_pow_collusion.py`               | Tests betrayal, breakup selection, and `gamma` race splitting.                                                 |
| `tests/test_plot_orphan_rate_comparison.py` | Tests summary loading, legacy path handling, and comparison-plot output.                                       |

## Run

Python 3 and NumPy are required. Plotting uses Matplotlib and SciencePlots;
Pandas and tqdm are optional helpers.

```bash
python3 pow_simulation.py --scenarios all
python3 pow_chain_withhold.py
python3 ocw_w_sweep.py
python3 pow_selfish.py
python3 pow_collusion.py
python3 pow_three_tolerate.py
python3 -m unittest discover -s tests -v
```

Use `python3 <file>.py --help` to see overrides and output paths.

---

# 中文简版

这是一个事件驱动的 PoW 仿真项目。出块服从 Poisson 过程，已发布区块默认瞬时传播。这样做是为了贴合零延迟理论模型并单独研究隐藏与发布策略。

默认实验使用可复现的 SHA-256 派生 seed。不同脚本分别使用固定时间或固定主链高度作为 horizon。

## 文件速记

### `pow_simulation.py`

这是固定时间 OCW 与 SM 的 DAA 对比主程序。场景 3 在 `p=0.65,0.75` 和 `n=1,2,3,5` 下使用 canonical DAA，场景 4 在相同 `p` 和 `n=1,3,10` 下使用 public/orphan-aware DAA。它对每个策略与 `p` 运行 10 次，并输出 raw runs、checkpoints、epoch statistics、summaries 和图。

### `pow_chain_withhold.py`

这个文件模拟连续链式隐藏策略。它扫描攻击者算力和 `w/T`。它输出主链份额、孤块率和图。

### `ocw_w_sweep.py`

这个文件专门扫描隐藏窗口。它把仿真点与理论曲线放在一起。图上的误差条是 95% 置信区间。

### `pow_selfish.py`

这个文件实现经典 Eyal--Sirer 自私挖矿基线。它支持 `p` 和 `gamma` 扫描。图上的误差条是标准误。

### `pow_collusion.py`

这个文件模拟三矿池 cartel。它包含合作、背叛、解体和容忍场景。它负责并行运行、汇总和绘图。

### `pow_three_tolerate.py`

这是三矿池容忍背叛场景的轻量入口。它复用 `pow_collusion.py` 的核心逻辑。只想快速跑该场景时使用它。

### `plot_orphan_rate_comparison.py`

这个文件不运行新仿真。它读取 OCW 和自私挖矿的汇总 CSV。它生成孤块率对比图和绘图数据。

### `try.py`

这是一个独立的公式试算脚本。它绘制 `p^2(1-p)/(1+p)`。它不参与正式实验。

### `tests/test_pow_simulation.py`

这个文件测试主实验的关键统计。它检查归一化、检查点、majority-mode 批量发布和 DAA。修改 `pow_simulation.py` 后应运行它。

### `tests/test_pow_collusion.py`

这个文件测试 cartel 的关键状态转换。它检查背叛、解体规则和 `gamma` 分流。修改合谋逻辑后应运行它。

### `tests/test_plot_orphan_rate_comparison.py`

这个文件测试孤块率对比工具。它检查 CSV 读取、旧路径兼容和图文件生成。修改绘图输入逻辑后应运行它。

## 快速运行

```bash
python3 pow_simulation.py --scenarios all
python3 pow_selfish.py
python3 pow_collusion.py
python3 -m unittest discover -s tests -v
```
