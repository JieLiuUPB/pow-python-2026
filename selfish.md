请你用 Python 编写一个“单个 selfish miner vs 其余 honest miners”的自私挖矿模拟程序，并保证代码可以直接运行。请严格按照下面要求实现，不要只给伪代码。

【建模目标】
系统中只有一个 self-mining 攻击者，其总算力占比为 p；其余所有矿工合并为 honest miners，总算力占比为 1-p。
模拟经典 Eyal-Sirer selfish mining 策略，而不是你自己发明的变体。请将 tie-breaking 参数 γ 作为可配置参数（默认 γ=0.5），表示当攻击者在“只领先 1 块、且 honest 刚追平”时，honest miners 中有 γ 的比例会跟随攻击者分支挖下一块。

【非常重要的严谨性要求】

1. 不要预设“selfish mining 的收益率一定大于 p”,但请注意p>0.5时，收益率一定大于p。
2. 正确做法是：对一组不同的 p 值进行模拟，输出每个 p 下攻击者收益率是否超过 p，并展示超过 p 的区间。
3. 代码中请明确区分：
   - 攻击者算力占比 p
   - 攻击者最终收益率 revenue_share
   - 系统孤块率 orphan_rate

【模拟策略】
请实现经典 selfish mining 状态机，核心逻辑应与 Eyal-Sirer 模型一致：

- 攻击者维护 private lead（私链领先主链的块数）
- honest 挖到块时：
  - 若攻击者没有领先，则直接接受 honest 块并推进公开主链
  - 若攻击者只领先 1，则攻击者立即公布该私块，进入竞争态（race/tie）
  - 若攻击者领先 2，则攻击者立即公布整条私链并确保主链替换
  - 若攻击者领先 >2，则攻击者只公布一部分私链以继续保持领先
- 攻击者自己挖到块时：
  - 若当前无竞争，则私链领先加 1
  - 若处于竞争态且攻击者再挖到块，则其竞争分支获胜，相应块计入主链
  - 其余状态按经典 selfish mining 状态机处理
    请你在代码注释中写清楚每个状态转移的含义。

【事件驱动模拟方式】
请采用离散事件模拟，而不是每秒钟循环一次。
推荐做法：

- 每一步只决定“下一块由谁先挖到”：selfish 的概率为 p，honest 的概率为 1-p
- 在 tie 状态下，如果下一块由 honest 挖到，则需要再根据 γ 决定该 honest 块是扩展攻击者分支还是 honest 分支
  这样做即可，无需模拟真实时间戳和指数分布等待时间；这里只关心区块发现顺序和主链/孤块统计。

【停止条件】
每一次独立模拟的停止条件是：

- “最终主链长度达到 2016 个区块”时停止
  注意：
- 这里的 2016 指主链上的非孤块数
- 模拟过程中产生的所有孤块不计入这 2016，但必须统计

【重复实验要求】
对每个 p 值：

- 使用 100 个不同 random seed 进行独立重复模拟
- 输出均值和误差
  误差请至少给出：
- 标准差 std
- 标准误 standard error
  如你愿意，也可额外给出 95% CI

【参数扫描】
默认请扫描如下 p 值：
p_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
并将 γ 设为可调参数，默认 gamma = 0.5。
同时，请在主程序最前面集中定义这些参数，方便我日后修改。

【统计指标定义】
请严格按如下定义计算：

1. revenue_share
   = 攻击者最终进入主链的区块数 / 最终主链总区块数
   注意最终主链总区块数应接近或等于 2016（按停止条件实现）

2. orphan_rate
   = 全系统孤块总数 / （主链块数 + 孤块总数）

请同时保留原始计数：

- main_chain_blocks_selfish
- main_chain_blocks_honest

【输出内容】
请输出两类结果：

A. 终端表格 / pandas DataFrame
每个 p 一行，列至少包括：

- p
- gamma
- mean_revenue_share
- std_revenue_share
- se_revenue_share
- mean_orphan_rate
- std_orphan_rate
- se_orphan_rate
- mean_selfish_orphan_rate
- mean_honest_orphan_rate
- revenue_minus_p = mean_revenue_share - p

B. 图像
至少画两张图：

1. p vs mean_revenue_share，并画出 y=x 参考线
2. p vs mean_orphan_rate
   若方便，也可以加误差棒（standard error 或 95% CI）

【代码质量要求】

1. 必须给出完整可运行 Python 代码
2. 使用：
   - random 或 numpy.random
   - pandas
   - matplotlib
3. 代码应拆分为清晰函数，例如：
   - simulate_one_run(p, gamma, seed, target_main_chain_blocks=2016)
   - run_experiments(p_list, gamma, n_repeats=100, target_main_chain_blocks=2016)
   - summarize_results(...)
   - plot_results(...)
4. 写足够多的注释，特别是 selfish mining 状态机转移逻辑
5. 保证随机种子可复现

【额外要求】
请在代码最后加一个 main 入口，使我直接运行脚本即可得到：

- 控制台表格
- 保存到本地的 csv 文件
- 保存到本地的 png 图

如果你认为某些状态机细节容易出错，请你优先保证“逻辑正确、可验证、便于修改”，而不是追求代码最短。
