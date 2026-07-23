# Qwen2.5-VL 四版本 Confidence 评测

本目录实现 Qwen2.5-VL-7B-Instruct 的四版本多模态 Confidence 批量评测：

- V1：visible previous confidence
- V2：hidden previous confidence
- V3：re-answer then confidence
- V4：full evidence baseline

正式数据集使用当前仓库的：

```text
datasets/dataset_test.json
```

不要使用已经删除的 `datasets/cleaned/dataset_test.json`。

## 1. 环境要求

需要 Python、PyTorch、Transformers、Qwen vision utility、Pillow 和 pytest：

```bash
pip install torch transformers qwen-vl-utils pillow pytest
```

模型必须已经下载到本地，默认路径为：

```text
qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct
```

模型约 16 GiB。建议使用显存充足的 GPU；当前实现会优先使用 BF16，失败时回退 FP16。

## 2. 最小真实运行

先运行 1 个 item、1 个 prior 验证环境和输出：

```bash
python -m confidence_test.four_version_evaluation \
  --model-path /root/autodl-tmp/qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct \
  --dataset datasets/dataset_test.json \
  --output-dir confidence_test/output \
  --variants all \
  --conditions all \
  --item-limit 1 \
  --prior-limit 1 \
  --overwrite
```

成功时会生成：

```text
confidence_test/output/v1_results.json
confidence_test/output/v1_simplified.json
confidence_test/output/v2_results.json
confidence_test/output/v2_simplified.json
confidence_test/output/v3_results.json
confidence_test/output/v3_simplified.json
confidence_test/output/v4_results.json
confidence_test/output/v4_simplified.json
```

## 3. 完整运行

运行全部 item、全部 prior、全部版本和全部 condition：

```bash
python -m confidence_test.four_version_evaluation \
  --model-path /root/autodl-tmp/qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct \
  --dataset datasets/dataset_test.json \
  --output-dir confidence_test/output \
  --variants all \
  --conditions all
```

默认参数：

```text
--variants all
--conditions all
--max-answer-tokens 24
--max-confidence-tokens 12
--output-dir confidence_test/output
--inference-path qwen-2.5-vl/inference.py
```

## 4. CLI 参数说明

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--model-path` | `qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct` | 本地 Qwen2.5-VL 模型目录。必须包含完整 checkpoint；不会联网下载。 |
| `--dataset` | `datasets/dataset_test.json` | 输入数据集 JSON。相对图像路径相对于该 JSON 所在目录解析。 |
| `--output-dir` | `confidence_test/output` | 完整结果和简化结果目录。必须位于 `confidence_test/` 内。 |
| `--inference-path` | `qwen-2.5-vl/inference.py` | 原始 `QwenVLInference` 源文件路径，用于动态导入模型加载逻辑。 |
| `--variants` | `all` | 要运行的版本，可用 `v1`、`v2`、`v3`、`v4`、完整版本名或 `all`；支持逗号分隔。 |
| `--conditions` | `all` | 要运行的图像条件，可用六种 condition 名称或 `all`；支持逗号分隔。 |
| `--max-answer-tokens` | `24` | Answer 阶段最多生成的 token 数。数值越大，允许的回答越长，显存和耗时也可能增加。 |
| `--max-confidence-tokens` | `12` | Confidence 阶段最多生成的 token 数。confidence 通常是短标签，不建议设置过大。 |
| `--item-limit` | 无限制 | 只处理数据集原始顺序的前 N 个 item，用于 smoke test 或小规模调试。 |
| `--prior-limit` | 无限制 | 每个 item 只处理前 N 个 `selected_text_priors`。 |
| `--overwrite` | 关闭 | 清空本次 `--variants` 选择的版本结果和简化结果后重新运行；不会删除或修改未选择版本。 |

### `--variants` 示例

```bash
# 只运行 V1
--variants v1

# 运行 V1 和 V3
--variants v1,v3

# 使用完整名称
--variants v2_hidden_previous_confidence

# 四个版本全部运行
--variants all
```

### `--conditions` 示例

```bash
# 只运行 null 和 irr
--conditions null,irr

# 只运行两种 conflict 条件
--conditions conflict_easy,conflict_hard

# 六种条件全部运行
--conditions all
```

### `--overwrite` 说明

不带 `--overwrite` 时，程序会读取已有的 `v*_results.json`，已完成的 stage/condition 会跳过，失败项默认重试：

```bash
python -m confidence_test.four_version_evaluation \
  --variants v3 \
  --conditions all
```

带 `--overwrite` 时，只重置当前选择的版本：

```bash
python -m confidence_test.four_version_evaluation \
  --variants v3 \
  --conditions all \
  --overwrite
```

上面的命令会重置 V3，但保留 V1、V2、V4 文件。V3 仍可以只读复用 V1/V2/V3 中已有且一致的公共文本阶段，不会修改其他版本文件。

## 5. 选择版本和 condition

只运行某个版本：

```bash
python -m confidence_test.four_version_evaluation \
  --variants v1 \
  --conditions all
```

支持的版本别名：

```text
v1
v1_visible_previous_confidence
v2
v2_hidden_previous_confidence
v3
v3_reanswer_then_confidence
v4
v4_full_evidence_baseline
all
```

只运行指定 condition：

```bash
python -m confidence_test.four_version_evaluation \
  --variants v3 \
  --conditions null,consistent_easy,conflict_hard
```

六种 condition 固定为：

```text
null
irr
consistent_easy
consistent_hard
conflict_easy
conflict_hard
```

## 6. 断点续跑

一个 checkpoint case 是：

```text
item × selected_text_prior
```

当前 prior 的所有已选择版本和 condition 完成后才写入磁盘，不会在每个 stage 后高频写入。

直接重新运行同一命令即可恢复：

```bash
python -m confidence_test.four_version_evaluation \
  --variants all \
  --conditions all
```

恢复规则：

- 已完成 stage 跳过；
- failed condition 默认重试；
- skipped condition 在依赖恢复后重新判断；
- V1/V2/V3 的公共文本阶段可从已有版本结果只读复用；
- `--overwrite` 只重置当前选择的版本，不影响其他版本。

例如只重新运行 V3：

```bash
python -m confidence_test.four_version_evaluation \
  --variants v3 \
  --conditions all \
  --overwrite
```

## 7. 限制 item 和 prior 数量

`--item-limit` 取数据集原始顺序的前 N 个 item；`--prior-limit` 对每个 item 取前 N 个 prior：

```bash
python -m confidence_test.four_version_evaluation \
  --variants all \
  --conditions all \
  --item-limit 10 \
  --prior-limit 2
```

## 8. 测试和语法检查

语法检查：

```bash
python -m py_compile \
  confidence_test/prompt_utils.py \
  confidence_test/runtime_imports.py \
  confidence_test/inference_extension.py \
  confidence_test/confidence_extension.py \
  confidence_test/answer_metrics.py \
  confidence_test/dataset_utils.py \
  confidence_test/io_utils.py \
  confidence_test/four_version_evaluation.py \
  confidence_test/tests/test_four_version_evaluation.py
```

单元测试不加载真实模型：

```bash
pytest -q \
  -o cache_dir=confidence_test/.pytest_cache \
  confidence_test/tests/test_four_version_evaluation.py
```

测试覆盖模型单实例、38 次阶段调用、公共阶段复用、六条件、V1/V2/V3/V4 依赖、答案概率、归一化 entropy、单行简化 JSON、断点续跑和 case 级 checkpoint。

## 9. 日志和结果

运行日志：

```text
confidence_test/logs/evaluation.log
```

日志记录：

- 当前 item/prior/version/condition/stage；
- 是否从 checkpoint 跳过；
- stage 耗时和错误类型；
- 累计模型调用次数；
- 输出文件路径。

不会记录完整 prompt、完整 logits、messages 或图像二进制。

完整 JSON 保存详细 answer/confidence 指标；简化 JSON 只保存：

```text
[answer, answer_prob, answer_entropy, soft_confidence]
```

失败 condition 始终保留为：

```json
[null, null, null, null]
```

## 10. 理论调用数量

全部版本和六种 condition 时，每个 `item × prior`：

```text
公共 Stage 1：1
公共 Stage 2：1
V1 Stage 3：6
V2 Stage 3：6
V3 Stage 3：6
V3 Stage 4：6
V4 Stage 1：6
V4 Stage 2：6
总计：38
```

整个进程只创建一个模型实例和一个 processor 实例。

## 11. 常见问题

### Processor 或模型加载时被 SIGKILL

检查 GPU 和 cgroup 内存：

```bash
python -c "import torch; print(torch.cuda.is_available())"
free -h
```

Qwen2.5-VL-7B checkpoint 约 16 GiB；如果 GPU 不可见且容器内存限制低于模型需求，真实 smoke test 无法完成，需要换到显存/内存充足的运行环境。

### 图像找不到

路径必须相对于 dataset 文件解析。使用默认数据集时，图像应位于：

```text
datasets/images/
```

### 只运行 V4

V4 不依赖公共文本阶段：

```bash
python -m confidence_test.four_version_evaluation \
  --variants v4 \
  --conditions all
```
