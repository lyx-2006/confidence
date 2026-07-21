# 颜色先验池生成工具

本项目为颜色问答建立五档文本先验池，并使用本地 Qwen2.5-VL 评估 Stage 1 答案和 Stage 2 confidence。

新增脚本：

- [confidence_analysis.py](/root/autodl-tmp/confidence_analysis.py)：独立执行 Stage 2 confidence 分析。
- [generate_color_pool.py](</root/autodl-tmp/generate color pool/generate_color_pool.py>)：执行 prior 筛选、DeepSeek 候选生成、稳定性测试和颜色池保存。

不会修改 `qwen-2.5-vl/inference.py`。

## 环境准备

本地模型必须存在：

```text
qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct
```

安装依赖：

```bash
pip install -r qwen-2.5-vl/requirements.txt
pip install openai
```

颜色池生成需要 API key：

```bash
export CSTCLOUD_API_KEY="你的 API key"
# 或 export OPENAI_API_KEY="你的 API key"
```

固定使用 `https://uni-api.cstcloud.cn/v1` 的 `DeepSeek-V4-Flash`。

## 独立 Confidence 分析

`--answer` 必须是 Stage 1 实际生成的答案。

```bash
python confidence_analysis.py \
  --question "What is the color of the square? Choose from: red, orange, yellow, green, blue, cyan, purple, pink, brown, white, black, gray." \
  --text-clue "A ripe lemon commonly has this color." \
  --answer yellow
```

保存 JSON：

```bash
python confidence_analysis.py \
  --question "What is the color of the square? Choose from: red, blue." \
  --text-clue "A ripe tomato commonly has this color." \
  --answer red \
  --output outputs/confidence_result.json
```

该脚本使用 assistant prefill `**confidence**:`，读取十类 confidence 的首 token logits，在十类之间 softmax 后计算 soft confidence；不收集 hidden state 或 attention。

Confidence 参数：

| 参数 | 含义 |
| --- | --- |
| `--question` | 问题，必填。 |
| `--text-clue` | 文本线索，必填。 |
| `--answer` | Stage 1 实际答案，必填。 |
| `--model-path` | 模型路径，默认 `qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct`。 |
| `--inference-path` | 原推理文件，默认 `qwen-2.5-vl/inference.py`。 |
| `--output` | JSON 输出路径；不指定则打印到终端。 |
| `--max-new-tokens` | Stage 2 最大生成 token 数，默认 `12`。 |

## 颜色池运行

默认增量模式：读取 `/root/autodl-tmp/datasets/dataset.json`，补齐缺失的 `color × bin`，输出 `datasets/color_prior_pool.json`。

```bash
python "generate color pool/generate_color_pool.py"
```

首次建议先运行 find：

```bash
python "generate color pool/generate_color_pool.py" --find --resume
```

find 顺序固定为：提取已有 prior → 本地模型测试 → 接纳合格 prior → 完成全部 find → 调用 DeepSeek 补齐缺失档位。find 完成前 DeepSeek 调用数为 0。

小规模试运行：

```bash
python "generate color pool/generate_color_pool.py" \
  --find --colors yellow --round 1 --target-per-bin 1 --resume
```

指定颜色：

```bash
python "generate color pool/generate_color_pool.py" --colors yellow,red,blue
```

`--after` 不包含参数本身的颜色：

```bash
python "generate color pool/generate_color_pool.py" --after yellow  # 从 green 开始
python "generate color pool/generate_color_pool.py" --after gray    # 从 maroon 开始
```

## 颜色池参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--find` | 关闭 | 先测试输入数据已有 prior；全部 find 完成前禁止 DeepSeek。 |
| `--after COLOR` | 无 | 从全局颜色顺序中该颜色的下一个颜色开始。 |
| `--round N` | `5` | 每个颜色最多执行的 Generator–Analyzer 轮数。 |
| `--input PATH` | `/root/autodl-tmp/datasets/dataset.json` | 输入数据集。 |
| `--output PATH` | `datasets/color_prior_pool.json` | 主结果 JSON。 |
| `--target-per-bin N` | `5` | 每个颜色、每个 bin 至少保留的 prior 数量。 |
| `--bin-batch-sizes A,B,C,D,E` | `30,30,20,10,20` | Bin 0 到 Bin 4 每轮候选数。 |
| `--deepseek-workers N` | `5` | DeepSeek 最大并发数；本地 Qwen 始终串行。 |
| `--colors A,B,C` | 全部颜色 | 只处理逗号分隔的颜色子集。 |
| `--resume` | 关闭 | 显式启用恢复语义；默认增量模式同样读取已有结果。 |
| `--seed N` | `42` | 问题选择、模板改写等确定性随机种子。 |
| `--near-duplicate-threshold X` | `0.88` | 近重复文本过滤阈值，越高越严格。 |
| `--stability-threshold X` | `0.1` | 稳定性阈值，严格使用 `soft_range < X`。 |

## Bin 与稳定性规则

| Bin | soft confidence 范围 | 默认每轮候选数 |
| --- | --- | ---: |
| 0 | `[0.0, 0.2)` | 30 |
| 1 | `[0.2, 0.4)` | 30 |
| 2 | `[0.4, 0.6)` | 20 |
| 3 | `[0.6, 0.8)` | 10 |
| 4 | `[0.8, 1.0]` | 20 |

Bin 0、Bin 1 每轮必须分别包含 10 条 `multi_step_reasoning`、10 条 `not_exclusion`、10 条 `pure_hard`。

每条 prior 在三个不同 shape 问题上测试：三次答案必须都是目标颜色，三次 Stage 2 必须成功，且 `max(soft_values) - min(soft_values) < stability_threshold`，三次平均值必须落入目标 bin。首测失败立即停止后续两题。问题不足时，会从同一 `Choose from:` 集合的真实问题模板替换 shape，并记录 `question_source: "template_rewrite"`。

## 输出与磁盘保护

每完成一个颜色，才使用临时文件和 `os.replace()` 原子更新：

```text
datasets/color_prior_pool.json
generate color pool/output/color_prior_generation_report.json
generate color pool/output/color_prior_generation_events.jsonl
generate color pool/output/color_prior_generation.log
generate color pool/output/color_prior_prompt_history.json
```

不会按候选、推理或轮次写 checkpoint。中断时当前颜色不会落盘，最多重做当前颜色；已完成颜色不会删除或覆盖。

## 注意事项

- 完整运行需要 GPU 和较大显存。
- 建议先用单颜色、单轮、`--target-per-bin 1` 验证。
- 缺少 `openai`、API key 或网络时，DeepSeek 阶段会明确报错，不会伪造结果。
- `--find` 和生成流程都会运行大量本地推理。
