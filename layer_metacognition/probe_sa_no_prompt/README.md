# Conflict-only No-SA Probe

本包研究：在 prompt 没有要求模型输出 Source Attribution 时，PTNL、PIT、AC、LAT、PANL
hidden state 是否已经含有与最终 SA target 相关的可线性解码信息。

## 数据隔离

- joint V4 `answer_basis_9` 只提供 `parsed_label`、`soft_image_score` 和答案。
- no-SA V4 `none/baseline` 只提供 answer-only hidden states。
- 两侧严格按 `item_id + prior_index + condition + version` 连接；不能按 case ID 连接。
- 本实验只接受 `conflict_easy`、`conflict_hard`。
- 主 cohort 是答案 normalization 后相等的 `answer_matched`；`all_joined` 是敏感性分析。
- split artifact 使用已有 `group_key=item_id` assignment，不重新随机划分。

## PANL

PANL 是 Answer 字段之后的 newline-bearing token，不是 SAC。真实模型预检必须确认
每个 completed case 都有 PTNL/PIT/AC/LAT/PANL，PANL 属于 answer stage，且 LAT 在
PANL 之前。答案末 token 与换行融合时，locator 记录 fusion 并调整 LAT；不得使用
EOS、SAC 或合成向量替代。

## 指标

Hard Probe 保持 class-balanced L2 multinomial logistic regression。主指标为 pooled OOF
accuracy；balanced accuracy、macro F1、AUROC、per-class support 和 confusion matrix
只作为诊断。Hard onset 是 Probe accuracy 相对 outer-train 多数类 OOF baseline 的
提升，其 item-bootstrap 95% CI 下界连续两层大于 0。Soft Probe 使用
StandardScaler + Ridge，主指标为 pooled OOF Spearman。Bootstrap 始终按 item 抽样。

附加 R² 分析只使用 `answer_matched` conflict-only cohort 和已有 item folds。Hard
label 通过 joint config 中的 `source_attribution_classes` / `source_attribution_midpoints`
转换为真实区间中点，单独训练 StandardScaler + Ridge；soft-score 直接复用已有 Ridge
OOF prediction。两者均计算 pooled OOF R² 和 item-bootstrap 95% CI，负 R² 不截断。

onset 是描述性的“可解码信号首次稳定高于统计基线”，不表示 causal formation。

## Resume 与产物

Probe 输出包括 `run_config.json`、`progress.json`、`join_manifest.json`、
`join_records.jsonl`、`split_audit.json`、`input_failures.jsonl`、
`unmatched_answers.jsonl`、`predictions/oof_predictions.jsonl`、按 cohort 的结果、
accuracy/Spearman 图、`onset.json` 和 `summary.json`。输出目录受保护；resume 会校验
immutable config 及 joint/no-SA/split/hidden-index SHA256，并只补缺失 prediction。

在已有正式 Probe 上只补充 R²（不会重新 capture hidden states，也不会覆盖已有指标）：

```bash
python -u -m layer_metacognition.probe_sa_no_prompt.analyze_no_sa_probe_results \
  --output-dir layer_metacognition/output/Final_v4_run_no_sa/baseline/stage_no_sa_prediction_probe \
  --r2-only
```

## 命令

真实 capture、PANL preflight、smoke、正式 Probe 和 nohup 串行命令见
[`layer_metacognition/README.md`](../README.md) 的 Conflict-only No-SA 部分。

单独执行 Probe 的统一入口：

```bash
python -u -m layer_metacognition.probe_sa_no_prompt.run_no_sa_prediction_probe \
  --joint-experiment-dir layer_metacognition/output/Final_v4_run_sa_prediction/answer_basis_9 \
  --no-sa-experiment-dir layer_metacognition/output/Final_v4_run_no_sa/baseline \
  --split-assignments layer_metacognition/output/Final_v4_run_sa_prediction/answer_basis_9/stage_sa_prediction_probe/split_assignments.json \
  --output-dir layer_metacognition/output/Final_v4_run_no_sa/baseline/stage_no_sa_prediction_probe \
  --device auto
```

No-SA Probe 阳性表示：没有显式 SA 输出要求时，hidden state 已包含与最终 SA 相关的
可线性解码信息。它不表示该位置已经因果决定 SA，或 SA 在该位置完成形成。
