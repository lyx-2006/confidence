# Hidden State SA Prediction Probe

本实验使用 MLLM 在不同 decoder layer、不同 token position 的 hidden state，预测最终 Source Attribution（SA）：

- Hard target：最终生成结果中的 `generated.source_attribution.parsed_label`，固定 9-class：`0...8`。
- Soft target：最终生成结果中的 `generated.source_attribution.soft_image_score`，连续值 `0...1`。
- 默认位置：`AC / LAT / PANL / SAC`。
- 默认 layer：`10 12 14 16 18 20 22 24 26 27`。
- 数据划分：基于 `item_id` 的 5-fold OOF；同一 item 不会同时进入 train 和 validation。

## 1. 前置 source run

Probe 不会重新运行 SA scorer，也不会加载 Qwen。它直接读取 source run 已保存的：

```text
config.json
results.jsonl
hidden_states/index.json
hidden_states/target_layers_*/shard_*.pt
```

如尚未生成 source artifact，运行：

```bash
python -u -m layer_metacognition.run_v3_v4_source_experiment \
  --output-dir layer_metacognition/output/Final_v4_run_sa_prediction \
  --versions v4 \
  --attribution-mode joint \
  --source-prompt-variant answer_basis_9 \
  --conditions all \
  --analysis_mode LMhead \
  --skip-attention \
  --skip-layer-readout \
  --skip_confidence \
  --save_hidden_state 10 12 14 16 18 20 22 24 26 27 \
  --save_hidden_state_positions ac lat panl ltt ptnl pit sac
```

Source run 中断后，使用相同命令并添加：

```bash
--resume
```

该命令的 source experiment 目录为：

```text
layer_metacognition/output/Final_v4_run_sa_prediction/answer_basis_9/
```

## 2. 正式运行 Probe

一次完成 OOF 训练、指标汇总、CSV 和轨迹图：

```bash
python -u -m layer_metacognition.probe_sa_prediction.run_sa_prediction_probe \
  --experiment-dir layer_metacognition/output/Final_v4_run_sa_prediction/answer_basis_9 \
  --output-dir layer_metacognition/output/Final_v4_run_sa_prediction/answer_basis_9/stage_sa_prediction_probe \
  --device auto
```

`--device auto` 会优先使用 CUDA 训练 hard-label Logistic Regression；CUDA 不可用时自动回退到 CPU。Soft-score Ridge Regression 使用 CPU。

## 3. 20-item smoke test

`--max-samples` 限制的是 `item_id` group 数量，并保留所选 item 的全部 eligible cases：

```bash
python -u -m layer_metacognition.probe_sa_prediction.run_sa_prediction_probe \
  --experiment-dir layer_metacognition/output/Final_v4_run_sa_prediction/answer_basis_9 \
  --output-dir layer_metacognition/output/Final_v4_run_sa_prediction/answer_basis_9/stage_sa_prediction_probe_smoke \
  --max-samples 20 \
  --device auto
```

建议使用独立 smoke 输出目录，避免与正式全量结果混合。

## 4. Resume

运行中断后，使用完全相同的配置和输出目录，并添加 `--resume`：

```bash
python -u -m layer_metacognition.probe_sa_prediction.run_sa_prediction_probe \
  --experiment-dir layer_metacognition/output/Final_v4_run_sa_prediction/answer_basis_9 \
  --output-dir layer_metacognition/output/Final_v4_run_sa_prediction/answer_basis_9/stage_sa_prediction_probe \
  --device auto \
  --resume
```

Resume 会扫描已有 OOF 唯一键，只补充缺失预测，不重复写入已完成 case。不可变配置或 source artifact 指纹发生变化时会拒绝 resume。

## 5. 分开运行训练与分析

只训练 probe：

```bash
python -u -m layer_metacognition.probe_sa_prediction.train_sa_probes \
  --experiment-dir layer_metacognition/output/Final_v4_run_sa_prediction/answer_basis_9 \
  --output-dir layer_metacognition/output/Final_v4_run_sa_prediction/answer_basis_9/stage_sa_prediction_probe \
  --device auto
```

基于已有 OOF prediction 重新生成结果、表格与轨迹图：

```bash
python -u -m layer_metacognition.probe_sa_prediction.analyze_sa_probe_results \
  --output-dir layer_metacognition/output/Final_v4_run_sa_prediction/answer_basis_9/stage_sa_prediction_probe
```

## 6. 常用参数

```text
--layers 10 12 ... 27       要分析的 zero-based decoder layer
--positions ac lat panl sac 要分析的 hidden-state position
--n-splits 5               item-level OOF fold 数量
--seed 42                  split 和 hard probe 随机种子
--max-samples N            最多选择 N 个 item_id group
--device auto|cuda|cpu     hard probe 训练设备
--resume                   从已有输出继续
```

指定部分 layer/position 的示例：

```bash
python -u -m layer_metacognition.probe_sa_prediction.run_sa_prediction_probe \
  --experiment-dir layer_metacognition/output/Final_v4_run_sa_prediction/answer_basis_9 \
  --output-dir layer_metacognition/output/Final_v4_run_sa_prediction/answer_basis_9/stage_sa_prediction_probe_subset \
  --layers 18 22 26 27 \
  --positions ac panl sac \
  --n-splits 5 \
  --seed 42 \
  --device auto
```

请求的 layer 和 position 必须已经存在于 source run 的 hidden-state index 中。

## 7. 输出

```text
stage_sa_prediction_probe/
  run_config.json
  progress.json
  split_assignments.json
  input_failures.jsonl
  results/
    hard_label_results.json
    soft_score_results.json
  predictions/
    oof_predictions.jsonl
  summary.json
  tables/
    layer_position_summary.csv
  plots/
    hard_label_layer_trajectory.png
    soft_score_layer_trajectory.png
```

主要文件：

- `oof_predictions.jsonl`：每个 task、position、layer、fold、case 的 OOF prediction。
- `hard_label_results.json`：accuracy、balanced accuracy、macro F1、macro one-vs-rest AUROC，以及 per-class AUROC/support。
- `soft_score_results.json`：Pearson、Spearman、MAE、R2，以及未裁剪预测的越界比例。
- `layer_position_summary.csv`：40 个 position-layer 组合的 pooled OOF 指标。
- `summary.json`：最佳组合、完整结果表和 H1/H2/H3 描述性分析。
- `progress.json`：completed/invalid job 数量和每个 fold 的 item leakage audit。
- `hard_label_layer_trajectory.png`：各位置随 layer 变化的 pooled OOF accuracy。
- `soft_score_layer_trajectory.png`：各位置随 layer 变化的 pooled OOF Spearman correlation。

`summary.json` 中的 hypothesis analysis 是方向性 OOF evidence。Hidden state 的线性可解码性不代表该表示对最终 SA 有因果作用。

## 8. 快速检查

运行完成后可检查：

```bash
python - <<'PY'
import json
from pathlib import Path

stage = Path(
    "layer_metacognition/output/Final_v4_run_sa_prediction/answer_basis_9/"
    "stage_sa_prediction_probe"
)
progress = json.loads((stage / "progress.json").read_text())
summary = json.loads((stage / "summary.json").read_text())

print("status:", progress["status"])
print("completed jobs:", progress["completed_job_count"])
print("invalid jobs:", progress["invalid_job_count"])
print("OOF predictions:", progress["prediction_count"])
print("best hard:", summary["best_hard"])
print("best soft:", summary["best_soft"])
PY
```

默认完整配置应得到：

```text
400 jobs = 2 tasks × 4 positions × 10 layers × 5 folds
40 hard position-layer results
40 soft position-layer results
```
