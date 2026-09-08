# Perturbation-based Real SA 与 verbal SA 的关系分析

本目录提供一个独立、CPU-only 的描述性分析。它只读取
`dp_SA/real_sa/output/tables/real_sa_per_case.csv`，不会修改 Real SA 实验逻辑或既有输出。

分析使用 `verbal_sa` 与 `G_R` 均存在且有限的全部 case，不按 `R_I_eligible` 或
`sign_type` 筛选。首先按 `signed_verbal_sa = 2 * verbal_sa - 1` 转换；这里
`G_R = phi_I - phi_T`；正值表示图像相对文本的贡献更大。

两个带截距的一元 OLS 回归分别回答：

1. `signed_verbal_sa = intercept_a + slope_b * G_R`：Real SA 能否解释 signed verbal SA；
2. `G_R = slope_a * signed_verbal_sa + intercept_b`：signed verbal SA 能否估计 Real SA。

每个模型报告样本内 $R^2$、Pearson、Spearman、MAE、斜率和截距，并以 seed 42
进行 2000 次 family-cluster bootstrap，给出百分位法 95% CI。每次 bootstrap
按 family 有放回抽样，在抽出的全部 case 上重新拟合并重新计算所有指标。

这里的 $R^2$ 是同一数据上的描述性样本内拟合度，不是样本外预测性能。一元带截距
OLS 中，两个回归方向的 Pearson、Spearman 与 $R^2$ 理论上相同，但回归系数和
MAE 不同。结果只能说明统计关系，不能单独证明 verbal SA 忠实反映了模型的真实
因果依赖。

双子图的两个坐标系都绘制 `x=0` 与 `y=0` 虚线，便于判断两个有符号指标所在象限。

运行：

```bash
python -m dp_SA.real_sa.relationship_analysis.run
```

输出位于本目录的 `output/tables/`、`output/artifacts/` 和 `output/figures/`。
