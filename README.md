# 颜色先验池生成工具

本项目为颜色问答建立五档文本先验池，并使用本地 Qwen2.5-VL 评估 Stage 1 答案和 Stage 2 confidence。

Layer Metacognition 的 V3/V4 Source Attribution、逐层 readout 和逐 Head
Attention Sink 使用说明见：[layer_metacognition/README.md](/root/autodl-tmp/layer_metacognition/README.md)。

新增脚本：

- [confidence_analysis.py](/root/autodl-tmp/confidence_analysis.py)：独立执行 Stage 2 confidence 分析。
- [generate_color_pool.py](/root/autodl-tmp/data_generation/legacy/generate_color_pool/generate_color_pool.py)：执行 prior 筛选、DeepSeek 候选生成、稳定性测试和颜色池保存。
- [test_deepseek_connection.py](/root/autodl-tmp/test_deepseek_connection.py)：低并发独立诊断 DeepSeek API 超时。

数据集生成代码统一放在 [data_generation/](/root/autodl-tmp/data_generation/)：共享的
V2 producer 与运行时（`generation_v2.py`、`generation_runtime.py`）、测试
（`tests/`）以及 legacy 生成器（`legacy/generate_dataset/`、
`legacy/generate_color_pool/`）。

不会修改 `qwen-2.5-vl/inference.py`（right-padding 修复只在
`confidence_test/inference_extension.py` 的扩展推理类中设置
`tokenizer.padding_side = "left"`）。

## 环境准备

本地模型必须存在：

```text
qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct
```

## Generation V2

V2 已拆为两个独立入口，共享同一个 Qwen batch 运行时
（`generation_runtime.py`：父进程唯一模型实例、逐 batch 串行、按 run root
哈希命名的持久化临时队列）：

- 文本（五档 entropy 颜色池）：`data_generation/legacy/generate_color_pool/generate_color_pool.py`，
  见下方「颜色池运行（V2 Entropy 单入口）」。
- 图像（shape-color 数据集）：`data_generation/legacy/generate_dataset/generate_shape_color_dataset.py`，
  不传 `--recreate`/`--legacy`/`--dry-run` 时即走 V2 图像 producer。

文本池 schema 为 `text_entropy_pool.v2`，五档 entropy score 为 0–100；每条
候选固定进行三次 text-only Qwen 测试。图像数据 schema 为
`shape_color_dataset.v2`，每个 easy/hard 结果都是带 `variant_index` 的数组，
文件名形如 `{id}_{branch}_{difficulty}({variant_index})`。正式产物固定写入
`generation_v2_outputs/formal/`（text/ 与 image/ 两个子目录）；中间文件
（Qwen job 队列等）写入按 run root 哈希命名的系统临时目录，不落输出目录。
Qwen 单次 batch 默认 `4` 个测试 job（`--qwen-batch-size`），避免 24 GB 显存
OOM。

图像入口示例（`--similarity-model-path` 与 `--download-similarity-model`
二选一）：

```bash
python data_generation/legacy/generate_dataset/generate_shape_color_dataset.py \
  --similarity-model-path /path/to/facebook-dinov2-base
# 或显式下载：--download-similarity-model
```

DINOv2 默认要求本地 `--similarity-model-path`；只有显式指定
`--download-similarity-model` 才下载 `facebook/dinov2-base` 到
`generation_v2_outputs/models/facebook-dinov2-base`。恢复运行必须使用
`--resume`，并保持 branch、数量、旋转、旧图及模型配置一致。

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

诊断脚本关闭 OpenAI SDK 自动重试，并输出每个请求的耗时、异常类型和响应长度。主颜色池程序使用 `timeout=150` 秒和 `max_retries=0`，由自己的三次重试逻辑统一记录。服务端高峰期响应可达 100s 以上（实测单请求 115.8s），120s 超时会把慢响应误判为失败，故超时放宽到 150s。单并发成功而五并发失败，通常表示并发过高或服务端限流；单并发也超时，则优先检查 API endpoint、Token 权限或服务状态。

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

## 颜色池运行（V2 Entropy 单入口）

该脚本已是 V2 五档 text-entropy 池的独立入口：`main()` 直接调用
`generation_v2.TextEntropyProducer`，与图像入口共用同一个 Qwen batch
运行时（`generation_runtime.py`：唯一模型实例、逐 batch 串行）和持久化
队列（位于系统临时目录）。
完整参数表格见
[data_generation/legacy/generate_color_pool/README.md](/root/autodl-tmp/data_generation/legacy/generate_color_pool/README.md)。
旧 confidence `PoolBuilder` 及其 confidence 版 prompt 已拆到同目录的
`legacy_pool_builder.py`（`from generate_color_pool import PoolBuilder` 仍可用），
CLI 不再调用它。

默认增量模式：读取 `/root/autodl-tmp/datasets/dataset.json`，为每个
`颜色 × entropy bin` 生成/补齐先验，输出
`generation_v2_outputs/formal/text/text_entropy_pool.json`。

```bash
python "data_generation/legacy/generate_color_pool/generate_color_pool.py"
```

`--find` 已废弃（V2 总是先验证再入队）。小规模试运行：

```bash
python "data_generation/legacy/generate_color_pool/generate_color_pool.py" \
  --colors red --round 1 --target-per-bin 1
```

指定颜色与熵档：

```bash
python "data_generation/legacy/generate_color_pool/generate_color_pool.py" --colors red,blue
python "data_generation/legacy/generate_color_pool/generate_color_pool.py" --select_pool 0,1,4   # 只生成最低两档和最高档
python "data_generation/legacy/generate_color_pool/generate_color_pool.py" --select_pool 40-80   # 等价的 score 区间写法
```

未被 `--select_pool` 选中的档位不进行 DeepSeek 候选生成。

`--after` 按 12 色顺序（red, orange, yellow, green, blue, cyan, purple, pink,
brown, white, black, gray）跳过之前的颜色：

```bash
python "data_generation/legacy/generate_color_pool/generate_color_pool.py" --after yellow  # 从 green 开始
```

V2 下 DeepSeek 调用由 producer 串行执行（每个 (颜色, bin) 的轮次独立），
Qwen 测试通过父进程唯一 scheduler 逐 batch 执行。`--deepseek-workers`、
`--color-workers`、`--stability-threshold` 仅为旧 CLI 兼容保留，不再生效。

## 颜色池参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--find` | 关闭 | 已废弃；V2 总是先验证再入队。 |
| `--after COLOR` | 无 | 从 12 色顺序中该颜色的下一个颜色开始。 |
| `--round N` | `5` | 每个 (颜色, bin) 最多执行的 Generator/Analyzer 轮数。 |
| `--input PATH` | `/root/autodl-tmp/datasets/dataset.json` | 输入数据集。 |
| `--output PATH` | `generation_v2_outputs/formal/text/text_entropy_pool.json` | 主结果 JSON。 |
| `--target-per-bin N` | `5` | 每个颜色、每个 entropy bin 至少保留的 prior 数量。 |
| `--select_pool BINS` | `all` | 需要生成的 entropy 档；支持 `0,1,4` 或 0–100 区间写法（如 `40-80`）。 |
| `--bin-batch-sizes A,B,C,D,E` | `20,20,20,20,20` | Bin 0 到 Bin 4 每轮 DeepSeek 候选数。 |
| `--deepseek-workers N` | `27` | 已失效（V2 串行调用 DeepSeek）；仅旧 CLI 兼容。 |
| `--color-workers N` | `6` | 已失效（V2 每颜色/bin 独立轮次）；仅旧 CLI 兼容。 |
| `--colors A,B,C` | 全部 12 色 | 只处理逗号分隔的颜色子集。 |
| `--resume` | 关闭 | 显式启用恢复语义；默认增量模式同样读取已有结果。 |
| `--seed N` | `42` | 问题选择、模板改写等确定性随机种子。 |
| `--near-duplicate-threshold X` | `0.88` | 近重复文本过滤阈值，越高越严格。 |
| `--stability-threshold X` | `0.1` | 已失效（V2 使用 entropy 档位判定）；仅旧 CLI 兼容。 |
| `--qwen-batch-size N` | `4` | 单次 Qwen batch 的测试 job 数（避免 24 GB 卡 OOM）。 |

## Entropy bin 与验收规则

| Bin | entropy_score 范围 | 默认每轮候选数 |
| --- | --- | ---: |
| 0 | `[0, 20)` | 20 |
| 1 | `[20, 40)` | 20 |
| 2 | `[40, 60)` | 20 |
| 3 | `[60, 80)` | 20 |
| 4 | `[80, 100]` | 20 |

`entropy_score` 基于受限 12 类颜色答案空间，以自然对数计算 entropy，再除以
`ln(12)` 映射到 0–100。

每个 (颜色, bin) 独立执行轮次：DeepSeek Generator 生成 `--bin-batch-sizes`
个候选 → 文本契约校验（shape-independent、按档位禁止具体颜色词等）→
DeepSeek Analyzer 判定 → 每个候选在三个不同形状问题上做 3 次 text-only
Qwen 测试。通过条件：三次答案必须都是目标颜色、restricted top-1 全是目标
颜色、三次熵实测落在同一档且 `max-min` 小于该档的容差（按档缩放
`5 + 2.5 × bin_id`：bin 0 为 5.0、bin 1 为 7.5、bin 2 为 10、bin 3 为 12.5、
bin 4 为 15）。容差按档缩放是因为熵波动主要由三个不同形状问题的形状竞争
引起，并随线索模糊度增长（强线索约 0.5、中等约 8–10、弱约 15–30），固定
`5.0` 会让 bin 2–4 几乎无法通过。按三次实测熵所在档归档（允许跨档路由：
为 Bin 0 生成但实测落入 Bin 1 的候选会写入 Bin 1；跨档时按实测档的容差
复核）。

## 每档 DeepSeek prompt

每个 entropy bin 的 DeepSeek Generator/Analyzer prompt 都是独立、特异化的
模板，保存在 `data_generation/prompts/text_entropy_bin_prompts.json`
（schema `text_entropy_bin_prompts.v1`，generator/analyzer 各 5 档）。可直接
编辑该 JSON 调优某一档的线索风格或判定标准，无需改代码。各档策略：

| Bin | 生成策略 | 推理步数 |
| --- | --- | ---: |
| 0 | 确定性：必须直接提及目标颜色词，陈述一目了然的常识事实 | 0–1 |
| 1 | 强指向：不点名颜色，指向日常物品/场景，读者第一联想 | 1 |
| 2 | 中等模糊：场景/季节/文化联想，需存在至少一个竞争猜测 | 1–2 |
| 3 | 高模糊：双重联想或多义物体，多步推理后才可辩护 | 2–3 |
| 4 | 极高模糊：隐晦/间接/悖论式联想，初看多个颜色都合理 | 3+ |

加载时会校验 schema、档位齐全和占位符完整（generator 模板支持
`{color}`、`{colors}`、`{count}`、`{accepted_json}`；analyzer 支持
`{color}`、`{bin_id}`、`{candidate_json}`），文件缺失或格式错误会明确报错，
不会静默改变生成行为。各档词项契约（bin 0 允许且要求目标色词、bins 1–4
禁止任何具体颜色词）仍由代码侧 `validate_color_lexical_contract` 强制。

## 输出与磁盘保护

每个 (颜色, bin) 每轮有 prior 入库时，才使用临时文件和 `os.replace()` 原子
更新主结果：

```text
generation_v2_outputs/formal/text/text_entropy_pool.json
```

不会按候选、推理或轮次写 checkpoint；Qwen 持久化队列和 worker 中间文件位于
按 run root 哈希命名的系统临时目录，不写入输出目录。

DeepSeek 请求会实时打印到终端：请求开始、成功解析（含响应字符数）、失败
原因和重试信息；这些实时日志不会触发 checkpoint 写入，也不会打印 API key
或完整 prompt。中断时当前 (颜色, bin) 不会落盘，最多重做当前条目；已完成
条目不会删除或覆盖。每个 (颜色, bin) 的每一轮结束后也会立即打印
`[TextEntropy] color=... bin=... round=N/M accepted=... (+N this round)`，
用于确认该轮测试结果。

## 注意事项

- 完整运行需要 GPU 和较大显存；`--qwen-batch-size` 默认 `4`，避免 24 GB
  卡在 eager attention 下大 batch OOM。
- 建议先用单颜色、单轮、`--target-per-bin 1` 验证。
- 缺少 `openai`、API key 或网络时，DeepSeek 阶段会明确报错，不会伪造结果。
