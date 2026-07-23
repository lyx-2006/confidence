# 颜色先验池生成工具

本项目为颜色问答建立五档文本先验池，并使用本地 Qwen2.5-VL 评估 Stage 1 答案和 Stage 2 confidence。

新增脚本：

- [confidence_analysis.py](/root/autodl-tmp/confidence_analysis.py)：独立执行 Stage 2 confidence 分析。
- [generate_color_pool.py](</root/autodl-tmp/generate color pool/generate_color_pool.py>)：执行 prior 筛选、DeepSeek 候选生成、稳定性测试和颜色池保存。
- [test_deepseek_connection.py](/root/autodl-tmp/test_deepseek_connection.py)：低并发独立诊断 DeepSeek API 超时。

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

固定使用 `https://uni-api.cstcloud.cn/v1` 的 `deepseek-v4-flash`，单次响应上限为 `2048` tokens。

## DeepSeek 超时诊断

先用单请求、单并发测试，不会加载 Qwen 或写入颜色池：

```bash
python test_deepseek_connection.py --repeat 1 --concurrency 1 --timeout 120
```

如果单并发成功，再测试并发是否导致服务限流或容量不足：

```bash
python test_deepseek_connection.py --repeat 5 --concurrency 5 --timeout 120
```

诊断脚本关闭 OpenAI SDK 自动重试，并输出每个请求的耗时、异常类型和响应长度。主颜色池程序也使用 `timeout=120` 秒和 `max_retries=0`，由自己的三次重试逻辑统一记录。单并发成功而五并发失败，通常表示并发过高或服务端限流；单并发也超时，则优先检查 API endpoint、Token 权限或服务状态。

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

运行 `--find` 时终端会显示 `[Find] started`、每条 prior 的 accepted/bin/reason、每个颜色的 bin 汇总，以及 `[Find] completed all colors`。

小规模试运行：

```bash
python "generate color pool/generate_color_pool.py" \
  --find --colors yellow --round 1 --target-per-bin 1 --resume
```

指定颜色：

```bash
python "generate color pool/generate_color_pool.py" --colors yellow,red,blue
```

一次可以选择 6 个颜色；它们按最多 3 色的同步 cohort 执行：

```bash
python "generate color pool/generate_color_pool.py" \
  --colors red,orange,yellow,green,blue,cyan \
  --color-workers 6
```

颜色级并行只作用于 DeepSeek 文本生成和 Analyzer。每个低档 bin 使用 3 个专属 Generator 和 3 个一一对应的 Analyzer；其他 bin 各使用 1 对。三个颜色同时处理 Bin 0、Bin 1 时，每个远端阶段会同步发出 `3 colors × 2 bins × 3 agents = 18` 个请求。默认选择全部 5 个 bin，峰值为 `3 × (2×3 + 3×1) = 27`，因此默认使用 27 路 DeepSeek 并发。Generator 完成后，本地 Qwen 仍按 candidate 串行评测；三个颜色会在 cohort barrier 等齐，随后同步发出对应的 18 个低档 Analyzer 请求。下一组颜色在前一 cohort 完成后开始。

只生成指定 confidence bins：

```bash
# 使用 bin 编号，只生成最低两档和最高档
python "generate color pool/generate_color_pool.py" --select_pool 0,1,4

# 等价的区间写法
python "generate color pool/generate_color_pool.py" \
  --select_pool 0.0-0.2,0.2-0.4,0.8-1.0
```

未被 `--select_pool` 选中的档位不会创建 Generator 或 Analyzer，也不会进行 DeepSeek 候选生成。`--find` 仍会测试数据集已有 prior，但只为所选档位补充生成内容。

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
| `--select_pool BINS` | `all` | 需要生成的 bins；支持 `0,1,4`、`bin0,bin1,bin4` 或区间写法。 |
| `--bin-batch-sizes A,B,C,D,E` | `40,40,20,10,40` | Bin 0 到 Bin 4 每轮候选数。 |
| `--deepseek-workers N` | `27` | DeepSeek 最大并发数；低档每 bin 按 3 个智能体计数，其他档按 1 个计数。只选两个低档且并发 3 色时至少为 18；全选时至少为 27。 |
| `--color-workers N` | `6` | 一次调度的颜色数，范围 `1-6`；实际按最多 3 色的同步 cohort 执行，本地 Qwen 始终串行。 |
| `--colors A,B,C` | 全部颜色 | 只处理逗号分隔的颜色子集。 |
| `--resume` | 关闭 | 显式启用恢复语义；默认增量模式同样读取已有结果。 |
| `--seed N` | `42` | 问题选择、模板改写等确定性随机种子。 |
| `--near-duplicate-threshold X` | `0.88` | 近重复文本过滤阈值，越高越严格。 |
| `--stability-threshold X` | `0.1` | 稳定性阈值，严格使用 `soft_range < X`。 |

## Bin 与稳定性规则

| Bin | soft confidence 范围 | 默认每轮候选数 |
| --- | --- | ---: |
| 0 | `[0.0, 0.2)` | 40 |
| 1 | `[0.2, 0.4)` | 40 |
| 2 | `[0.4, 0.6)` | 20 |
| 3 | `[0.6, 0.8)` | 10 |
| 4 | `[0.8, 1.0]` | 40 |

默认 40 条时，Bin 0、Bin 1 使用三个互相隔离的 Generator–Analyzer 对：`prior_knowledge_agent` 负责 15 条 `prior_knowledge_multistep`（类似 “The color has the same color as a morpho butterfly's wings”）以及剩余 5 条 `free_form`；`not_exclusion_agent` 负责 10 条显式使用 `not` 并否定其他候选颜色的 `not_exclusion`；`high_difficulty_agent` 负责 10 条 `high_difficulty`。每个 Analyzer 只读取所属 Generator、所属源 bin 的评测结果和跨档结果，并独立维护下一轮 prompt。自定义 batch size 时仍按 `15:10:10:5` 等比例调整。

Bin 0、Bin 1 的 prompt 强调“目标颜色仍是唯一最佳答案，但证据较弱并保留多个可信替代项”，避免为了降低 confidence 而让答案本身发生变化。Bin 4 强调直接、典型、无歧义的常识关联，避免模糊、否定、冷门事实和竞争答案。Bin 2、Bin 3 的策略与批量保持不变。

每条 prior 在三个不同 shape 问题上测试：三次答案必须都是目标颜色，三次 Stage 2 必须成功，且 `max(soft_values) - min(soft_values) < stability_threshold`。Bin 0、Bin 1 生成的候选即使首测越出目标档也会继续完成三问；稳定后按照三次 `soft_mean` 的实际区间归档，例如为 Bin 0 生成但实测均值为 `0.35` 的数据会写入 Bin 1。其他档位仍要求落入其生成目标档，首测越界会立即停止。问题不足时，会从同一 `Choose from:` 集合的真实问题模板替换 shape，并记录 `question_source: "template_rewrite"`。

## 输出与磁盘保护

每完成一个颜色，才使用临时文件和 `os.replace()` 原子更新：

```text
datasets/color_prior_pool.json
generate color pool/output/color_prior_generation_report.json
generate color pool/output/color_prior_generation_events.jsonl
generate color pool/output/color_prior_generation.log
generate color pool/output/color_prior_prompt_history.json
```

不会按候选、推理或轮次写 checkpoint。

DeepSeek 请求会实时打印到终端：请求开始、成功解析（含响应字符数）、失败原因和重试信息；这些实时日志不会触发 checkpoint 写入，也不会打印 API key 或完整 prompt。中断时当前颜色不会落盘，最多重做当前颜色；已完成颜色不会删除或覆盖。
每个颜色的每一轮结束后也会立即打印 `[ColorPool] color=... round=... accepted=... missing_bins=... color_complete=...`，用于确认该轮是否正常完成。

## 注意事项

- 完整运行需要 GPU 和较大显存。
- 建议先用单颜色、单轮、`--target-per-bin 1` 验证。
- 缺少 `openai`、API key 或网络时，DeepSeek 阶段会明确报错，不会伪造结果。
- `--find` 和生成流程都会运行大量本地推理。
