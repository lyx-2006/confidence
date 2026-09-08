# Confidence gap 与双 confidence 的冻结测试集比较

这是一个独立、CPU-only 的拟合实验，只读取 `dp_SA/unimodal_logit_confidence/output/results/`
内既有冻结数据，不修改原实验逻辑或历史输出。

## 定义

使用 Phase 0 fixed answer 的校准单模态 confidence：

\[
V_{SA}=2\,soft\_SA-1,\quad L_i=logit(C_i),\quad L_t=logit(C_t),\quad G_L=L_i-L_t.
\]

概率在取 logit 前裁剪至 `[1e-6, 1-1e-6]`。比较两个带截距模型：

- `M_G`：`V_SA = beta_0 + beta_G * G_L`
- `M_IT`：`V_SA = beta_0 + beta_i * L_i + beta_t * L_t`
- `M_GM`：令 `M_L=(L_i+L_t)/2`，写成
  `V_SA = beta_0 + beta_G * G_L + beta_M * M_L`

不会把 `G_L`、`L_i`、`L_t` 同时放入一个模型，因为 `G_L=L_i-L_t` 会造成完全
线性冗余。`M_G` 检验 SA 是否主要像相对 confidence 比较器；`M_IT` 允许模型以不同
强度分别读取图像与文本 confidence。`M_GM` 与 `M_IT` 使用完全相同的信息，其中
`beta_M` 表示 confidence gap 不变时，整体 confidence 是否仍改变 SA。

为了在 Ridge 下保持二者真正等价，`M_GM` 是已拟合 `M_IT` 的精确代数重参数化，而
不是把重新标准化的 `G_L,M_L` 再独立拟合一次。后者会改变 Ridge 的惩罚几何，因而
不再保证预测相同。程序以 `1e-12` 为阈值审计二者逐 case 预测的一致性。

## 拟合与评估

- Train：冻结 1,112 条；Test：冻结 100 条、50 个 family；family/item 完全隔离。
- 预测变量的均值与标准差只在 train 计算；模型也只在 train 拟合。
- 两个模型均使用 `Ridge(alpha=1, solver=lsqr)`，与此前 `MC` 分析保持相同模型类型。
- Test 只用于一次 held-out 评估，报告 R²、Pearson、Spearman 与 MAE。
- `M_GM - M_G`（数值等于 `M_IT - M_G`）的 ΔR² 使用 seed 42、2,000 次 paired family-cluster bootstrap；
  每次对 test family 有放回抽样，两个模型共享同一抽样。

标准化系数表示相应预测变量增加一个 train 标准差时，预测的 `V_SA` 变化量。理论上的
简单 confidence 竞争预期 `beta_G>0`、`beta_i>0`、`beta_t<0`。本实验是冻结测试集上的
样本外统计比较，不能单独证明 confidence 对 SA 的因果作用。

运行：

```bash
python -m dp_SA.confidence_gap_comparison.run
```

输出包括模型指标、标准化系数、paired ΔR²、逐 case test 预测、预测对真实值双子图，
以及带 `M_IT` 预测等高线的二维 `(L_t,L_i)` 图。
