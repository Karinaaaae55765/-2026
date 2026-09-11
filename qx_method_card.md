# Q2 Method Card

## Goal and success criteria

在第一次全向源测向后，构造可保证再次获得有效示向度的第二检测点候选区域；选择一个第二检测点，使真实目标位置与测向误差处于最不利情形时，二次交会定位区域的最小包围圆半径最小。若最优半径不超过20m，可将包围圆圆心作为具有保证清除意义的目标位置。

## Human constraints

- Output form: 第二检测点选择策略、候选区域、可计算的最优点与定位半径。
- Priority: 鲁棒定位误差优先；移动距离仅作次级择优。
- Unacceptable failure: 选点后无法接收信号、返回near、交会区域无界，或以未声明的概率分布代替题设误差界。
- Experiment budget: Q2无具体数据，采用二维候选网格、场景离散和局部加密；完整数值预算由代码阶段根据运行时间确定。

## Shortlist

| ID | Role | Mathematical idea | Why eligible | Main risk | Implementation cost |
|---|---|---|---|---|---|
| M2-RR | main_candidate | 候选区域上的最小-最大鲁棒优化，以二次交会区域最小包围圆半径为损失 | 直接反映最坏定位误差，与20m清除半径可比，不需要误差分布 | 连续内层最坏情形和分段目标计算量较大，可能偏保守 | 中等 |
| M2-ANGLE | usable_baseline | 最大化所有可能目标下的最小交会角正弦，并以移动距离破同分 | 产生合法第二点，计算简单，解释透明 | 交会角是误差的代理量，不能完整反映不规则定位区域 | 低 |
| M2-SLACK | conditional_fallback | 当严格保证接收的候选域为空时，引入最大接收距离违约量并字典序最小化 | 保证模型在极端几何下仍能返回可执行点 | 不再保证所有可能目标均能在第二点被检测 | 中等 |

## Mathematical formulation

首测后目标不确定集合：

\[
\Omega_1=W_1\cap B(O,1800)\cap B(S_1,1500)\setminus B(S_1,5).
\]

鲁棒候选区域：

\[
\mathcal C=\left\{S_2:\max_{G\in\Omega_1}\|S_2-G\|_2\le1000,\quad
\min_{G\in\Omega_1}\|S_2-G\|_2>5\right\}.
\]

对场景 \(G\in\Omega_1\)、\(e_2\in[-1^\circ,1^\circ]\)，第二示向度为

\[
\alpha_2=\operatorname{atan2}(G_y-S_{2y},G_x-S_{2x})+e_2,
\]

二次交会区域为

\[
P_2(S_2;G,e_2)=\Omega_1\cap W_2(S_2,\alpha_2).
\]

最小包围圆半径：

\[
\rho(P)=\min_c\max_{X\in P}\|X-c\|_2.
\]

鲁棒目标：

\[
J(S_2)=\max_{G\in\Omega_1,\ e_2\in[-1^\circ,1^\circ]}\rho(P_2(S_2;G,e_2)),
\qquad
S_2^*=\arg\min_{S_2\in\mathcal C}J(S_2).
\]

若多个候选点的 \(J\) 差值不超过数值容差，则选择 \(\|S_2-S_1\|_2\) 较小者。

基线目标：

\[
S_{2,\mathrm{angle}}^*=\arg\max_{S_2\in\mathcal C}
\min_{G\in\Omega_1}\sin\beta(S_2,G).
\]

## Solver plan

1. 由首测信息构造并离散表示 Ω1。
2. 计算每个平面网格点到 Ω1 的最大、最小距离，得到候选区域 C。
3. 对候选区域粗网格取点；对每个点遍历覆盖 Ω1 与误差区间的场景。
4. 保留首测获得的全部物理约束，显式构造
   \(P_2=\Omega_1\cap W_2\)；圆域采用内、外接正多边形逼近并检查分辨率收敛。旧定义
   \(W_1\cap W_2\) 作废，不得用于正式评价。
5. 计算P2的最小包围圆半径，取全部场景中的最大值作为J。
6. 在最优粗网格附近逐层缩小步长，直至点位与目标值稳定。
7. 用最坏交会角基线计算另一个第二点，并以同一鲁棒半径J比较。

## Baseline validity

- Real task completed: 是；基线可输出候选域内一个第二检测点。
- Comparable output/metric: 是；将基线点代入同一J函数，可与主模型直接比较。

## Risk-probe summary

| ID | Executability | Data/assumptions | Degeneracy | Sensitivity | Scale | Verdict |
|---|---|---|---|---|---|---|
| M2-RR | 模型闭合，待实现 | 不需要误差分布；依赖首测为direction | 候选域空、交会无界需显式处理 | 需检查网格和场景密度 | 二维可用粗细网格控制 | CONDITIONAL |
| M2-ANGLE | 解析定义明确 | 同一不确定集 | 平行交会时指标趋近0 | 需检查目标距离端点 | 计算量低 | CONDITIONAL |

## Fallback trigger

- Trigger: 严格候选区域C为空，或在可接受网格分辨率下无点满足1000m保证接收约束。
- Evidence to evaluate: 候选域面积、最小可能最大距离、约束违约量。

## Compact history

- 2026-09-10：建模手提出并确认“构造候选区域后使用半径评估第二点，采用鲁棒优化策略”（decision_id: q2_robust_radius_method）。
