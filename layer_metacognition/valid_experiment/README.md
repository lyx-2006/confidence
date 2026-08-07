# Semantic Patchscope Prompt 鲁棒性验证

该目录提供独立的 Semantic Patchscope 验证实验，用于比较：

- Semantic Prompt 的五种同义措辞；
- 数字类别与来源语义反转；
- 类别行展示顺序乱序。

所有版本的 `soft_image_score` 都统一为 `0 → 更偏文本，1 → 更偏图像`。
实验固定使用 V3/V4 joint answer + Source Attribution 流程。每个 case 的
teacher-forced 多模态序列只执行一次 hooked forward，同时采集 AC 和 SAC；
不会运行 confidence layer readout、CC hidden state、entropy、attention、
Identity Patchscope 或 SAC LMhead。

## 文件

- `semantic_variants.py`：variant 定义、动态 Prompt 和固定结果列顺序。
- `run_semantic_patchscope_validation.py`：实验、增量保存和恢复。
- `analyze_validation_results.py`：raw/corrected 鲁棒性统计。
- `test_semantic_validation.py`：CPU 单元及本地 tokenizer 集成测试。

## 最小实验

```bash
python layer_metacognition/valid_experiment/run_semantic_patchscope_validation.py \
  --dataset datasets/datasets.json \
  --image-root datasets \
  --model-path qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct \
  --output-dir layer_metacognition/valid_experiment/output/smoke_test \
  --versions v3 \
  --attribution-mode joint \
  --conditions conflict_easy \
  --max-items 1 \
  --prior-indices 0 \
  --semantic-variants all
```

## 完整实验

```bash
python layer_metacognition/valid_experiment/run_semantic_patchscope_validation.py \
  --dataset datasets/datasets.json \
  --image-root datasets \
  --model-path qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct \
  --output-dir layer_metacognition/valid_experiment/output/full_validation \
  --versions v3 \
  --attribution-mode joint \
  --conditions all \
  --semantic-variants all
```

中断后在原命令末尾添加 `--resume`。恢复时会修复 JSONL 的尾部半行、验证
配置一致性，并从 `validation_details.jsonl` 重建两个主结果。若所有 case
已经完成，则不会重新加载模型。

单个 case 如果出现答案/来源格式无法解析、图片缺失或答案 token 冲突，会写入
`validation_failures.jsonl` 并继续下一个 case。模型加载、target 初始化、
hooked forward、Patchscope forward 或持久化错误仍会终止运行，以免掩盖系统性
故障。

## 主实验可选参数

```text
--dataset PATH
```

数据集 JSON 路径。默认：
`datasets/datasets.json`。

```text
--image-root PATH
```

可选的图片根目录。未提供时，图片相对路径以数据集文件所在目录为基准；
提供后，以该目录为基准重新解析相对图片路径。

```text
--model-path PATH
```

本地 Qwen 模型目录。默认：
`qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct`。

```text
--inference-path PATH
```

包含 `QwenVLInference` 的 Python 文件。默认：
`qwen-2.5-vl/inference.py`。

```text
--output-dir PATH
```

实验输出目录。默认：
`layer_metacognition/valid_experiment/output/validation`。

```text
--versions v3 [v4 ...]
```

选择 V3、V4 或两者，可使用逗号或空格分隔。固定执行顺序为 V3、V4。
默认：`v3`。示例：

```bash
--versions v3 v4
```

```text
--attribution-mode joint
```

来源归因模式。本实验为保证 AC/SAC 来自同一 teacher-forced forward，只支持
`joint`。默认：`joint`。

```text
--conditions CONDITION [CONDITION ...]
```

选择一个或多个图片条件，可使用逗号或空格分隔。可选值：

```text
null
irr
consistent_easy
consistent_hard
conflict_easy
conflict_hard
all
```

默认：`all`。输入 `all` 时不能同时输入单独 condition，实际执行顺序始终采用
上述固定顺序。

```text
--max-items N
```

只读取数据集展开前的前 `N` 个 item，必须为正整数。默认不限制。

```text
--item-ids ID [ID ...]
```

只运行指定 item ID，可使用逗号或空格分隔。默认运行所有已读取 item。
该筛选发生在 `--max-items` 之后。

```text
--prior-indices INDEX [INDEX ...]
```

只运行指定 prior 下标，可传多个非负整数。默认运行所有 prior。

```text
--semantic-variants VARIANT [VARIANT ...]
```

选择 Semantic Prompt variants，可使用逗号或空格分隔。默认：`all`。
可选值及固定结果顺序：

```text
base
synonym_reliance
synonym_evidential_weight
synonym_support_balance
synonym_evidence_distribution
synonym_dominant_support
reverse_direction
order_shuffle_1
order_shuffle_2
order_shuffle_3
all
```

自定义选择必须包含 `base`；`all` 不能与其他值混用；重复值会被拒绝。CLI
输入顺序不会影响运行和结果列顺序。例如：

```bash
--semantic-variants base reverse_direction order_shuffle_1
```

```text
--max-answer-tokens N
```

初始答案允许生成的最大 token 数，同时参与 joint generation 总预算计算。
必须为正整数，默认：`24`。

```text
--max-source-tokens N
```

来源归因部分预留的最大 token 数，同时参与 joint generation 总预算计算。
必须为正整数，默认：`4`。

```text
--resume
```

从已有 `validation_details.jsonl` 恢复。未指定时，如果输出目录已经存在
`config.json`，程序会拒绝覆盖。恢复时 dataset、模型、版本、conditions、
筛选条件、variants 和 token budgets 必须与保存配置完全一致。已经写入
`validation_failures.jsonl` 的 skipped case 也会被跳过，不会重复卡住。

## 输出

- `config.json`：数据选择、完整 variants、模型 runtime 与比较层范围。
- `target_baselines.json`：每个 Semantic target 的一次性 unpatched baseline。
- `validation_details.jsonl`：逐 case 的 restricted logits、概率和 raw/corrected
  Semantic 分数。
- `validation_failures.jsonl`：逐行记录被跳过的 sample-local 错误及阶段。
- `validation_results.json`：固定列的 raw `soft_image_score`。
- `validation_results_corrected.json`：固定列的 baseline-corrected 分数。
- `progress.json`：原子更新的运行进度。

校正分数按以下定义计算：

```text
delta_logits = patched_class_logits - baseline_class_logits
corrected_soft_image_score = softmax(delta_logits) · variant_image_midpoints
```

主结果的前两列始终为 `answer`、`answer_probability`，后续 variant 列始终
采用预定义顺序。主结果 layer 数组中的所有数值以三位小数写入；详细 JSONL
保留原始全精度。最后一个 transformer layer 会被保存，但不参与默认比较。

## 分析

```bash
python layer_metacognition/valid_experiment/analyze_validation_results.py \
  --input-dir layer_metacognition/valid_experiment/output/full_validation
```

默认累计变化层段为 18→24，可通过 `--analysis-start-layer` 和
`--analysis-end-layer` 修改。`validation_summary.json` 同时包含 raw 和
corrected 的逐层 MAE、Spearman、层间变化相关性、累计变化以及 group
population variance。Spearman 使用平均秩处理 ties；样本不足或常数序列输出
`null`。`reverse_direction` 还会额外输出 `1 - reverse` 与 Base 的逐层及
轨迹对齐指标，用于检验 hidden state 是否保持固定数字方向。

## 分析脚本参数

```text
--input-dir PATH
```

必填。必须包含 `config.json`、`validation_results.json` 和
`validation_results_corrected.json` 的实验输出目录。分析结果写入同一目录的
`validation_summary.json`。

```text
--analysis-start-layer N
--analysis-end-layer N
```

指定累计变化 `S[end] - S[start]` 的起止层，默认分别为 `18` 和 `24`。
两个层号都必须位于 `config.json` 记录的 comparison layer 范围内，且 start
必须小于 end；最后一个 transformer layer 不能作为分析层。

示例：

```bash
python layer_metacognition/valid_experiment/analyze_validation_results.py \
  --input-dir layer_metacognition/valid_experiment/output/full_validation \
  --analysis-start-layer 16 \
  --analysis-end-layer 24
```

## 测试

```bash
python -m unittest \
  layer_metacognition.valid_experiment.test_semantic_validation -v
```

单元测试不加载 7B 权重；本地 Qwen processor/tokenizer 存在时会验证真实
chat template、target 最后 token 和 raw digit 协议。完整 smoke test 需要可见
CUDA GPU。
