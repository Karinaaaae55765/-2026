# Q2 代码实现计划（round2）

## 唯一实现口径

- 数学定义唯一来源：项目根目录 `model_assumptions.md`。
- 决策编号：`q2_robust_radius_method`。
- 主方法 M2-RR：最小化离散场景下最坏的最小包围圆半径。
- 基线 M2-ANGLE：最大化最坏交会角正弦，再用同一半径指标复核。
- M2-SLACK 未激活；严格候选域为空时只报告触发证据。

第二次定位集合必须为

\[
P_2=\Omega_1\cap W_2,
\qquad
\Omega_1=W_1\cap B(O,1800)\cap B(S_1,1500)\setminus B(S_1,5).
\]

旧定义 `W1 ∩ W2` 作废。

## 数值实现

1. 分别用圆的内接、外接正多边形构造 `Omega1`；楔形由Q1叉积半平面精确裁剪。
2. 外接多边形包含连续圆域，用它验证候选点的1000m最大距离约束；对填满near孔洞的外包络检查最小距离，得到偏保守但安全的非near候选域。
3. 用全局网格与长轴法向对称种子共同生成候选点。
4. 对真实位置场景和含端点的误差场景，显式裁剪得到每个 `P2=Omega1_outer ∩ W2`。
5. 对P2顶点计算最小包围圆和旋转卡壳直径；主方法按最坏半径选择，近似同值时按移动距离选择。
6. 用粗、中、细三档圆边数、场景数量和搜索步长检查收敛。

候选接收约束具有外包络意义；对真实位置和误差的最大化仍是有限场景离散，因此结果不得表述为连续问题的解析全局最优。

## 输入与输出

距离单位为m，方位角按x轴正向逆时针计。默认算例 `S1=(0,0)`、首测角 `0°` 仅为合成演示，不是题目官方数据。

结果写入：

```text
results/Q2/experiments/round2/
├── figures/q2_candidate_region.png
├── figures/q2_objective_map.png
├── metrics/q2_metrics.json
├── tables/candidate_comparison.csv
├── tables/sensitivity.csv
├── tables/boundary_validation.csv
└── run_summary.json
```

默认终端输出中文摘要；`--json` 输出机器可读JSON。固定随机种子为2026。

## 必须通过的审查

- `syntax`：所有脚本可导入、可运行。
- `input_contract`：单位、半径、角度、计数和路径合法。
- `method_alignment`：每次P2均保留Omega1，不再使用作废定义。
- `candidate_feasibility`：最远距离不超过1000m，最小距离严格大于5m。
- `boundary_cases`：跨零角、误差端点、空候选域、圆域截断、旧定义无界回归。
- `reproducibility`：相同参数重复运行得到相同最优点与指标。
- `output_contract`：主方法、基线、灵敏度、边界表、图和运行摘要齐全。
