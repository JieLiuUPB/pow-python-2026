# pow_three_tolerate.py 快速理解笔记

## 1. 文件目标
- 这是一个“三矿池 BetrayTolerated 场景”的最小运行器，依赖 `pow_collusion.py` 的核心仿真逻辑。
- 默认只跑三矿池：`b, s, h`，其中 traitor 只能是 `b` 或 `s`。
- 输出每次 run 的完成信息，以及所有 run 的平均收益率。

## 2. 核心结构
- `FastCollusionSimulation(CollusionSimulation)`：
  - 覆盖 `_publish_public_block`，用于更快地直接发布公共区块。
  - 覆盖 `simulate_one_run`，在单进程模式下使用 `tqdm` 显示单次 run 的 canonical block 进度。
- `build_arg_parser()`：
  - 定义命令行参数，包括仿真参数、三矿池配置、随机种子、事件上限、进度输出粒度。
- `run_single(...)`：
  - 单次 run 的执行函数，用于串行模式和多进程并行模式复用。
- `validate_three_only(...)`：
  - 检查参数范围、算力和为 1、矿池 ID 完整性（必须有 `b,s,h`）等。
- `validate_progress_step_percent(...)`：
  - 约束进度输出步长在 `[1, 100]`。
- `validate_jobs(...)`：
  - 约束并行进程数 `jobs > 0`。
- `main()`：
  - 解析参数 -> 校验 -> 构造场景 -> 串行或并行执行 `runs` -> 汇总并打印均值收益率。
  - 实际并行度为 `min(jobs, runs)`，避免 `jobs` 大于 `runs` 时启动多余进程。
  - 多进程模式下由主进程使用 `tqdm` 显示整体 runs 完成进度。

## 3. 运行流程（主线）
1. 解析参数（`argparse`）。
2. 解析 `--three-pools` 得到三矿池算力配置。
3. 构造 `SimConfig`、`ScenarioConfig(BetrayTolerated)`。
4. 对每个 run：
   - 基于 `seed_base + run_id` 生成 seed。
   - 创建 `FastCollusionSimulation`。
   - 执行 `simulate_one_run(progress_step_percent=...)`。
   - 打印 run 完成信息。
5. 统计并打印 `b/s/h` 的平均收益率。

补充：当 `--jobs > 1` 且 `runs > 1` 时，使用 `ProcessPoolExecutor` 按 run 维度并行执行，属于多核模式。单个 run 本身不会再拆分成多个进程。

## 4. 进度输出机制
- 参数：`--progress-step-percent`（默认 `1`）。
- 单进程模式：
  - 使用 `tqdm` 显示当前 run 的 canonical block 进度条。
  - `progress_step_percent` 用来控制进度条的批量刷新粒度，减少频繁刷新开销。
- 多进程模式：
  - 不在子进程中显示每个 run 的进度条。
  - 由主进程显示一个总进度条，表示已完成的 runs 数量。
- 实现位置：覆盖后的 `simulate_one_run` 与 `main()` 中的 `tqdm`。

## 5. 关键参数速查
- `--T`：平均出块间隔相关参数（Poisson 过程尺度）。
- `--gamma`：竞态时诚实矿工偏向参数。
- `--runs`：重复实验次数。
- `--target-blocks-long`：每次 run 的目标主链长度（停止条件）。
- `--betray-on-nth-opportunity` / `--betray-start-height` / `--q` / `--betray-threshold`：背叛触发相关参数。
- `--three-pools`：三矿池算力配置，默认 `b=0.37,s=0.33,h=0.3`。
- `--three-traitor`：三矿池中的 traitor，默认 `s`。
- `--progress-step-percent`：进度条刷新粒度百分比，默认 `1`。
- `--jobs`：请求的并行进程数，默认 `30`。实际并行进程数为 `min(jobs, runs)`；若 `runs=1`，仍然只会执行单个 run。

## 6. 维护规则（必须遵守）
- 每次修改 `pow_three_tolerate.py` 后，必须同步更新本文件：
  - `核心结构`（新增/删除函数、类行为变化）
  - `运行流程`（执行路径变化）
  - `关键参数速查`（参数新增/默认值变化）
  - `变更记录`（追加一条）

## 7. 变更记录
- 2026-03-08：
  - 新增运行进度输出能力（`--progress-step-percent`，默认 10%）。
  - `FastCollusionSimulation` 覆盖 `simulate_one_run`，在事件循环中输出阶段进度。
  - 新增 `--jobs` 多进程并行支持（按 run 分发到多核）。
  - 新建本 Markdown 文档，作为快速理解与后续同步维护基准。
  - 调整并行 worker 数为 `min(jobs, runs)`，并在 `jobs > runs` 或 `runs=1` 时输出提示，减少无意义的进程池初始化。
  - 将进度展示改为 `tqdm`：单进程显示单 run 进度，多进程显示整体 runs 进度。
