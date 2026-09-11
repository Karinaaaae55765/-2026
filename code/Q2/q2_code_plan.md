# Q2 代码实现计划（round3）

## 唯一实现口径

- 数学定义来源：项目根目录 `model_assumptions.md`。
- 决策编号：`q2_conditional_reception_2026-09-11`。
- 第一次成功检测后的联合约束：`1000 <= R <= 1500` 且 `distance(S1,G) <= R`。
- 对每个目标样本的最小相容接收半径：`max(1000, distance(S1,G))`，不得再把候选约束固定为 1000 m。
- 第二次非近距定位集合：`P2 = Omega1 ∩ W2 ∩ B(S2,1500)`；旧版 `W1 ∩ W2` 作废。
- 第二点返回 `near` 时直接定位，场景损失为 0；`S2 == S1` 排除。

## 算法契约

1. 构造并采样 `Omega1`，同时保留圆域外接多边形用于定位区域裁剪。
2. 对搜索网格计算每个候选点到每个目标样本的距离与条件半径：
   - `signal_mask`：所有样本均满足第二距离不超过条件半径；
   - `bearing_mask`：在 `signal_mask` 基础上，所有样本距离严格大于 5 m。
3. 主方法与交会角基线必须使用同一个 `signal_mask`，不得分别改变可行域。
4. 对每个目标—误差场景：距离不超过 5 m 时损失为 0；否则构造新版 `P2` 并计算最小包围圆半径。
5. 主方法按离散最坏半径、移动距离和稳定坐标次序择优；基线按最坏交会角择优，再用同一半径函数评价。
6. 候选区域 CSV 必须逐点输出坐标、首测局部坐标、响应/示向标志、条件接收余量及距离范围。

## 输入与输出

距离单位为 m，方位角按 x 轴正向逆时针计。默认 `S1=(0,0)`、首测角 `0°` 是合成演示，可用命令行参数替换。

结果目录：

```text
results/Q2/experiments/round3/
├── figures/q2_candidate_region.png
├── figures/q2_objective_map.png
├── metrics/q2_metrics.json
├── tables/candidate_region.csv
├── tables/candidate_comparison.csv
├── tables/sensitivity.csv
├── tables/boundary_validation.csv
└── run_summary.json
```

默认终端只打印中文摘要；`--json` 输出机器可读 JSON。固定随机种子为 2026。round1、round2 作为历史证据保留，不覆盖。

## 必须通过的审查

- `syntax`：脚本可导入、可运行。
- `input_contract`：半径、角度、计数、路径和步长合法。
- `conditional_reception`：存在由条件规则恢复、但被旧固定 1000 m 规则错误删除的测试点。
- `candidate_regions`：分别输出非空的响应候选域和纯示向候选域，并排除同址。
- `near_branch`：第二点进入 5 m 内时按直接成功处理。
- `method_alignment`：非近距场景始终保留 `Omega1`，主方法和基线共享候选域。
- `boundary_cases`：跨零角、误差端点、圆域截断和旧定义无界回归均通过。
- `reproducibility`：相同参数重复运行的核心 JSON 指标一致。
- `output_contract`：中文摘要、完整候选点表、比较表、灵敏度、边界表、两张图和摘要齐全。

## 数值声明

候选域和内层最坏情形都采用确定性有限离散。CSV 给出的是网格近似，面积用网格点数乘单元面积估算；结果不得表述为连续候选域的解析边界或全局最优证明。
