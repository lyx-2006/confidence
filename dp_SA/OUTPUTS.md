# `dp_SA` 输出目录说明

本文档按实验目录整理 `dp_SA` 下所有 `output` 目录。路径均相对于仓库根目录 `/root/autodl-tmp`；批量生成的 case/shard 文件使用通配符表示，实际文件名可在对应目录中查看。

## 总体代码入口

| 实验 | 主要代码 |
|---|---|
| 通用 SA 分析、捕获、probe、选择、steering、分数计算 | `dp_SA/analysis.py`、`capture.py`、`probe.py`、`selection.py`、`steering.py`、`soft_score.py`、`run_pipeline.py` |
| LAT difficulty swap | `dp_SA/lat_difficulty_swap/` 下的脚本 |
| PANL information | `dp_SA/panl_information/` 下的脚本 |
| confidence steering | `dp_SA/confidence_steering/` 下的脚本 |
| unimodal logit confidence | `dp_SA/unimodal_logit_confidence/` 下的脚本 |
| answer-matched LAT steering | `dp_SA/answer_matched_lat_steering/` 下的脚本 |
| checkpoint steering | `dp_SA/checkpoint_steering/` 下的脚本 |
| real SA | `dp_SA/real_sa/` 下的脚本；关系分析见 `relationship_analysis/` |
| confidence gap comparison | `dp_SA/confidence_gap_comparison/` 下的脚本 |
| LAT/PANL/CLE/SAC 轨迹传输 | `dp_SA/SA_trajectory/` 下的 `LAT2PANL/`、`CLE_transport/` 脚本 |

## 1. `lat_difficulty_swap`

代码目录：`dp_SA/lat_difficulty_swap/`。该实验用 difficulty 相关方向进行 latent steering，比较不同 difficulty 处理对 SA/答案指标的影响。

```text
dp_SA/lat_difficulty_swap/output/
├── results/
│   ├── summary.md                         # 正式结果汇总
│   ├── artifacts/                         # candidate pair、clean 结果、审计、hidden（*.pt）和 figure data
│   ├── figures/                           # 正式图形结果
│   ├── progress/                          # 运行进度和配置
│   ├── reanalysis/                        # 再分析结果
│   └── tables/                            # 指标表
└── smoke_tmp/round_{1,2,3}/               # 多轮 smoke 结果
```

## 2. `panl_information`

代码目录：`dp_SA/panl_information/`。该实验分析 PANL 表征携带的信息及其与 SA 结果的关系。

```text
dp_SA/panl_information/output/
└── results/
    ├── summary.md                         # 正式结果汇总
    ├── artifacts/                         # capture、probe、校准、审计和 hidden（*.npz）
    ├── figures/                           # PANL 分析图形
    ├── progress/                          # 运行进度和配置
    └── tables/                            # PANL/probe 指标表
```

## 3. `answer_matched_lat_steering`

代码目录：`dp_SA/answer_matched_lat_steering/`。该实验进行 answer-matched 的 LAT steering，并和 PANL 方向做比较；`smoke` 目录仅用于快速检查运行状态。

```text
dp_SA/answer_matched_lat_steering/output/
├── results/{artifacts,figures,progress,tables}/ # 正式结果产物
├── results/summary.md                     # 正式 answer-matched LAT 结果
├── lat_panl_comparison/                   # LAT 与 PANL 比较（含 summary、artifacts、figures、progress、tables）
├── lat_panl_comparison_smoke/round_{1,2}/ # 比较实验 smoke 产物
└── smoke_tmp/round_{1,2}/                  # 主实验 smoke 产物
```

## 4. `checkpoint_steering`

代码目录：`dp_SA/checkpoint_steering/`。该实验在 checkpoint hidden 上进行 steering trial，并保留诊断、图形、进度和表格。

```text
dp_SA/checkpoint_steering/output/results/
├── artifacts/diagnostics/                  # clean capture、steering trials 和诊断汇总
├── artifacts/hidden/*.npz                  # case 级 hidden
├── figures/                                # steering 诊断图
├── progress/                               # 运行进度和配置
└── tables/                                 # steering 结果表
```

## 5. `confidence_gap_comparison`

代码目录：`dp_SA/confidence_gap_comparison/`。该实验比较 global confidence gap、image/text logit 等特征对 SA 结果的预测能力。

```text
dp_SA/confidence_gap_comparison/output/
├── artifacts/
│   ├── audit.json                          # 输入、划分和运行审计
│   └── test_predictions.csv                # 测试集预测
├── figures/
│   ├── fig1_test_predictions.png           # 测试预测图
│   └── fig2_confidence_plane.png           # confidence plane 图
└── tables/
    ├── model_performance.csv               # 各模型性能
    ├── paired_delta_r2.csv                 # 配对 R² 差异及 bootstrap 区间
    └── standardized_coefficients.csv       # 标准化回归系数
```

## 6. `confidence_steering`

代码目录：`dp_SA/confidence_steering/`。该实验沿 confidence-defined LAT 方向做 steering，并包含 Fast L14 复现、CPU 可解码性诊断、正交方向、自然尺度分解等分析。

```text
dp_SA/confidence_steering/output/
├── {results,all_fast_l14,all_fast_l14_smoke,cpu_target_diagnostic,orthogonal_results}/
│   ├── summary.md                          # 各配置汇总
│   ├── artifacts/                          # case 结果、hidden、审计和 bootstrap 产物
│   ├── figures/                            # 各配置图形
│   ├── progress/                           # 各配置进度和运行参数
│   └── tables/                             # 各配置结果表
└── natural_decomposition/
    ├── summary.md                          # confidence 自然尺度分解
    ├── gradient_validation_summary.md       # LAT -> PANL/SAC 局部梯度验证
    └── random_sa_subspace_null_summary.md  # 匹配随机 SA 子空间 null
    # 该目录同时包含 artifacts/、figures/、progress/、tables/
```

### 6.1 confidence steering 的 trajectory 子实验

代码目录：`dp_SA/confidence_steering/trajectory/`。该实验保存 confidence steering 的轨迹结果和完成状态。

```text
dp_SA/confidence_steering/trajectory/output/results/
├── CONSOLIDATION.md                         # 轨迹结果整合说明
├── README_RESULTS_zh.md                     # 中文结果说明
└── completion.json                          # 运行完成状态
```

`trajectory/output` 还包括 `audit_delivery/`（audit manifest、fingerprint、hidden reuse 审计）、`archive/`（归档位置）和 `smoke/` 下的多组历史 smoke 配置；`results/` 另含 `artifacts/`、`figures/`、`progress/`、`tables/` 及按 case 保存的 hidden/trajectory 文件。

### 6.2 confidence steering 的 robust check 子实验

代码目录：`dp_SA/confidence_steering/robust_check/`。该实验用于正式结果的稳健性复核。

```text
dp_SA/confidence_steering/robust_check/output/
├── formal_nohup.log                         # 正式运行日志
├── formal_nohup_resume.log                  # 续跑日志
└── results/
    ├── README_RESULTS_zh.md                 # 中文结果说明
    ├── completion.json                       # 运行完成状态
    ├── artifacts/figures/progress/tables/   # 稳健性复核产物
    └── *.jsonl                              # case 级结果和审计记录
```

## 7. `unimodal_logit_confidence`

代码目录：`dp_SA/unimodal_logit_confidence/`。该实验从 image/text unimodal logits 构造并校准 confidence，再训练 probe；`results/` 是正式产物，`smoke_tmp/` 是运行检查产物。

```text
dp_SA/unimodal_logit_confidence/output/
├── results/summary.md                        # 正式结果总览
├── results/shared/
│   ├── completion.json
│   ├── input_fingerprints.json
│   ├── run_config.json
│   ├── split_audit.json
│   └── manifests/*.jsonl                     # family、probe、calibration、test 划分清单
├── results/unimodal_confidence/
│   ├── artifacts/raw_scores/*.jsonl          # image/text unimodal 原始分数
│   ├── artifacts/calibrated_scores/*.jsonl   # 校准后的 unimodal 分数
│   ├── artifacts/predictions/*.jsonl         # phase-1 confidence 合并预测
│   ├── artifacts/temperature/*               # temperature 参数和搜索轨迹
│   ├── progress/*.json                       # 评分和温度拟合进度
│   └── tables/temperature_calibration.csv   # 温度校准表
├── results/confidence_probe/
│   ├── artifacts/hidden/shard_*/             # 各 shard/case 的 hidden state（.npz）
│   ├── artifacts/models/*.joblib             # 各 phase/layer 的 probe 模型
│   ├── artifacts/predictions/*.jsonl         # train/test probe 预测
│   ├── figures/probe_{pearson,r2,spearman}.png
│   ├── progress/*.json                       # capture、训练和分析进度
│   └── tables/probe_metrics*.csv             # probe 指标
├── results/explanatory_comparison/summary.md # 正式解释性比较汇总
└── smoke_tmp/
    ├── smoke_report.json                     # smoke 总报告
    └── round_1/{single_gpu,dual_gpu}/        # 单卡/双卡 smoke 的 hidden、manifest、raw score 和 progress
```

`smoke_tmp/round_1/*` 内的具体文件结构与正式结果相同，主要包括 `confidence_probe/artifacts/hidden/`、`confidence_probe/progress/`、`shared/manifests/` 和 `unimodal_confidence/artifacts/raw_scores/`。

## 8. `real_sa`

代码目录：`dp_SA/real_sa/`。该实验通过图像/文本模态破坏与 mean-embedding 基线估计答案依赖；它不是无条件的“真实模态使用比例”。

```text
dp_SA/real_sa/output/
├── README.md                                  # 实验定义、条件和解释边界
├── artifacts/
│   ├── condition_scores.jsonl                 # 各破坏条件的分数
│   └── parity_audit.jsonl                     # 条件 parity 审计
├── progress/
│   ├── analysis.json
│   ├── coverage_audit.json
│   ├── formal_report.json
│   ├── forward_budget.json
│   ├── processor_audit.json
│   └── run_config.json                         # 分析、覆盖率、预算和运行配置
└── tables/
    ├── corruption_diagnostics.csv             # 破坏诊断
    ├── real_sa_per_case.csv                   # case 级 real-SA
    ├── real_sa_summary.csv                    # real-SA 汇总
    └── sign_type_summary.csv                  # sign type 汇总
```

### 8.1 `real_sa/relationship_analysis`

代码目录：`dp_SA/real_sa/relationship_analysis/`。该子实验检验 verbal SA 与 gradient-based SA (`G_R`) 的双向关系。

```text
dp_SA/real_sa/relationship_analysis/output/
├── artifacts/predictions.csv                  # 关系模型预测
├── figures/fig1_bidirectional_relationship.png # 双向关系图
└── tables/
    ├── table1_verbal_sa_from_gr.csv           # 用 G_R 预测 verbal SA
    └── table2_gr_from_verbal_sa.csv           # 用 verbal SA 预测 G_R
```

## 9. `SA_trajectory`

代码目录：`dp_SA/SA_trajectory/`。该组实验研究 LAT、PANL、CLE、SAC 之间的轨迹/表示传输关系。

### 9.1 `LAT2PANL`

代码目录：`dp_SA/SA_trajectory/LAT2PANL/`。该实验将 LAT 方向与 PANL 方向进行轨迹/指纹级比较。

```text
dp_SA/SA_trajectory/LAT2PANL/output/
└── results/
    ├── fingerprint.json                       # 轨迹/输入指纹
    ├── artifacts/                             # 轨迹和 case 产物
    ├── figures/                               # 轨迹图
    ├── progress/                              # 运行进度
    └── tables/                                # 统计表
```

### 9.2 `CLE_transport`

代码目录：`dp_SA/SA_trajectory/CLE_transport/`。该实验检查 CLE 到 SAC、PANL 到 CLE 的传输运行。

```text
dp_SA/SA_trajectory/CLE_transport/output/
├── CLE2SAC/
│   ├── .gitkeep
│   ├── completion.json                        # CLE -> SAC 完成状态
│   ├── run_config.json                        # CLE -> SAC 配置
│   ├── artifacts/manifests/test_manifest.jsonl # 测试 manifest
│   ├── artifacts/trials/*.json                # 各 case、condition、layer 的 trial
│   ├── artifacts/trials.jsonl                 # trial 汇总
│   ├── figures/                               # 传输图
│   ├── progress/                              # 运行进度
│   ├── smoke/                                 # smoke 结果
│   └── tables/                                # 传输统计表
└── PANL2CLE/
    ├── .gitkeep
    ├── completion.json                        # PANL -> CLE 完成状态
    ├── run_config.json                        # PANL -> CLE 配置
    ├── statistics_1_to_4_completion.json      # statistics 1-4 完成状态
    ├── artifacts/manifests/test_manifest.jsonl # 测试 manifest
    ├── artifacts/trials/*.json                # 各 case、condition、layer 的 trial
    ├── artifacts/trials.jsonl                 # trial 汇总
    ├── figures/                               # 传输图
    ├── progress/                              # 运行进度
    ├── smoke/                                 # smoke 结果
    └── tables/                                # 传输统计表
```

## 路径核对

生成或更新本文档后，可用下面的命令检查文档中列出的具体路径是否仍存在；带 `*` 的批量模式需要在对应目录内按实际文件展开：

```bash
find dp_SA -type d -name output -print | sort
find dp_SA -type f -path '*/output/*' -print | sort
```
