# V3/V4 Source Attribution 与逐 Head Attention Sink

本目录用于运行 Qwen2.5-VL 的 V3/V4 来源归因实验。实验在回答与置信度之外，增加 Source Attribution（SA）、三种 SAC hidden-state 逐层读出，以及逐 attention head 的 source sink 分析。

新实验入口为：

```bash
python layer_metacognition/run_v3_v4_source_experiment.py
```

它不会修改旧的 `confidence_test/prompt_utils.py`、`layer_metacognition/run_main_experiment.py` 或旧 V1–V4 输出。

## 推荐：一次运行全部 mode

`--attribution-mode all` 已实现。它会在同一进程内依次执行：

```text
case 1: none → parallel → joint → 刷新简化 JSON
case 2: none → parallel → joint → 刷新简化 JSON
...
```

模型、processor 和 tokenizer 只加载一次；V3 的 initial answer/confidence 按 `item × prior` 生成一次，并在三个 mode 之间共享。

这里的一个 case 指同一个：

```text
item_id × prior_index × condition × version
```

程序不会先跑完整数据集的所有 `none`。每组 mode 完成后，都会原子刷新 `analysis_minimal.json`、`analysis_source_sink_minimal.json` 和 `summary.json`，然后才进入下一个 case。单条完整结果仍会优先 append、flush、fsync 到 `results.jsonl`，因此简化文件刷新失败或进程中断时，可通过 `--resume` 从 JSONL 重建。

单 item、单 prior、单 condition 的建议测试命令：

```bash
python layer_metacognition/run_v3_v4_source_experiment.py \
  --dataset datasets/dataset_with_images.json \
  --image-root datasets \
  --model-path qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct \
  --output-dir layer_metacognition/output/v3_v4_all \
  --versions v3 v4 \
  --attribution-mode all \
  --analysis_mode LMhead Identity Semantic \
  --conditions consistent_easy \
  --max-items 1 \
  --prior-indices 0
```

每个实验 case 会产生三个独立记录，`case_id` 的末尾分别为 `none`、`parallel` 和 `joint`。例如：

```text
1__prior_0__consistent_easy__v3__none
1__prior_0__consistent_easy__v3__parallel
1__prior_0__consistent_easy__v3__joint
```

如果只想运行一种 mode，把 `all` 换为 `none`、`parallel` 或 `joint`。

## 三种 mode 的流程

| mode | 生成流程 | SA / SAC |
| --- | --- | --- |
| `none` | 使用旧 V3/V4 answer 与 confidence 调用 | 不生成 SA，SAC 为空 |
| `parallel` | answer、SA、confidence 分别独立生成 | SA 与 answer 隔离 |
| `joint` | answer 与 SA 同次生成，confidence 后续独立生成 | AC 与 SAC 来自同一 assistant 序列 |
| `all` | 对每个 case 依次执行上述三个 mode，再进入下一 case | 同一输出目录保存全部结果 |

SA 输出必须严格为：

```text
**Source Attribution**:<CLASS>
```

`<CLASS>` 是 `0`–`8`，冒号后不能有空格。joint 输出必须严格包含两行：

```text
**Answer**: <ANSWER>
**Source Attribution**:<CLASS>
```

解析失败时会保留 raw output，并将 case 写为终态失败记录；程序不会用受限 logits 的 argmax 冒充已解析标签。

## 三种 SAC analysis mode

`--analysis_mode` 可同时选择一个或多个 SAC hidden-state 解码方法：

| analysis mode | 含义 |
| --- | --- |
| `LMhead` | 将当前层 SAC hidden state 直接经过 final norm 和 LM head，读取最终词表基底中的九类分布 |
| `Identity` | 将当前层 SAC hidden state patch 到同层的 content-free identity target，再经过剩余模型层读取九类分布 |
| `Semantic` | 将当前层 SAC hidden state patch 到同层的 content-free Source Attribution 语义 target，再经过剩余模型层读取九类分布 |

Identity 和 Semantic 表示“当前层 hidden state 经固定 target prompt 和剩余模型层可恢复出的分布”，不是“真实的中间层概率”。三种方法默认只运行 `LMhead`，保持旧行为。输入顺序不会改变执行和保存顺序，固定为：

```text
LMhead → Identity → Semantic
```

例如：

```bash
--analysis_mode Identity Semantic
--analysis_mode Semantic LMhead
--analysis_mode LMhead Identity Semantic
```

重复输入会报错。Identity 和 Semantic target 不包含具体 case 的 question、image、text clue、answer、confidence 或 condition；target inputs 和无 patch baseline 在同一进程中只准备、计算一次。

## 完整命令格式

```bash
python layer_metacognition/run_v3_v4_source_experiment.py \
  [--dataset DATASET] \
  [--image-root IMAGE_ROOT] \
  [--model-path MODEL_PATH] \
  [--inference-path INFERENCE_PATH] \
  [--output-dir OUTPUT_DIR] \
  [--versions {v3,v4} ...] \
  [--attribution-mode {none,parallel,joint,all}] \
  [--analysis_mode {LMhead,Identity,Semantic} ...] \
  [--conditions CONDITION ...] \
  [--max-items N] \
  [--item-ids ITEM_ID ...] \
  [--prior-indices INDEX ...] \
  [--resume] \
  [--skip-attention] \
  [--skip-layer-readout]
```

查看程序生成的最新参数帮助：

```bash
python layer_metacognition/run_v3_v4_source_experiment.py --help
```

## 参数说明

### 数据、模型和输出

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--dataset` | `datasets/dataset_with_images.json` | 数据集 JSON 路径 |
| `--image-root` | 不设置 | 相对图片路径的根目录；不设置时保持 dataset-relative 解析 |
| `--model-path` | `qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct` | 本地模型目录 |
| `--inference-path` | 项目内默认 `inference.py` | 现有 `QwenVLInference` 实现路径 |
| `--output-dir` | `layer_metacognition/output/v3_v4_source` | 输出目录；不同配置建议使用不同目录 |

### 实验范围

| 参数 | 默认值 | 可输入内容 |
| --- | --- | --- |
| `--versions` | `v3 v4` | `v3`、`v4`，可选一个或两个 |
| `--attribution-mode` | `none` | `none`、`parallel`、`joint`、`all` |
| `--analysis_mode` | `LMhead` | `LMhead`、`Identity`、`Semantic`，可选择一个或多个 |
| `--conditions` | `all` | `null`、`irr`、`consistent_easy`、`consistent_hard`、`conflict_easy`、`conflict_hard` 或 `all` |
| `--max-items` | 不限制 | 正整数；在 prior/condition 展开前限制原始 item 数 |
| `--item-ids` | 全部 | 一个或多个 item ID，支持空格或逗号分隔 |
| `--prior-indices` | 全部 | 一个或多个从 0 开始的 prior index |

### 运行控制

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--resume` | 关闭 | 从已有 `results.jsonl` 恢复；已有 `case_id` 不会重复运行 |
| `--skip-attention` | 关闭 | 跳过逐 head attention sink，减少显存和运行时间 |
| `--skip-layer-readout` | 关闭 | 跳过 AC/CC 以及全部三种 SAC 逐层读出和 Patchscope baseline |
| `--max-answer-tokens` | `24` | answer 最大生成 token 数 |
| `--max-confidence-tokens` | `12` | confidence 最大生成 token 数 |
| `--max-source-tokens` | `4` | parallel SA 最大生成 token 数 |

所有选项均使用 `--参数 值`。布尔开关不接值，例如使用 `--resume`，不要写 `--resume true`。

## 常用命令

只运行 V4 parallel：

```bash
python layer_metacognition/run_v3_v4_source_experiment.py \
  --versions v4 \
  --attribution-mode parallel \
  --conditions consistent_easy consistent_hard \
  --max-items 1 \
  --prior-indices 0 \
  --output-dir layer_metacognition/output/v4_parallel
```

运行全部数据和全部 condition：

```bash
python layer_metacognition/run_v3_v4_source_experiment.py \
  --versions v3 v4 \
  --attribution-mode all \
  --conditions all \
  --output-dir layer_metacognition/output/v3_v4_all
```

运行 V4 parallel 的三种 SAC 分析：

```bash
python layer_metacognition/run_v3_v4_source_experiment.py \
  --dataset datasets/dataset_with_images.json \
  --image-root datasets \
  --model-path qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct \
  --output-dir layer_metacognition/output/v4_sac_three_modes \
  --versions v4 \
  --attribution-mode parallel \
  --analysis_mode LMhead Identity Semantic \
  --conditions consistent_easy \
  --prior-indices 0 \
  --skip-attention
```
完整运行
```bash
python layer_metacognition/run_v3_v4_source_experiment.py \
  --dataset datasets/dataset_with_images.json \
  --image-root datasets \
  --model-path qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct \
  --output-dir layer_metacognition/output/sac_analysis_mode_run \
  --versions v4 \
  --attribution-mode parallel \
  --analysis_mode LMhead Identity Semantic \
  --conditions consistent_easy \
  --max-items 1 \
  --prior-indices 0 \
  --skip-attention
```

跳过高开销分析，只检查生成流程：

```bash
python layer_metacognition/run_v3_v4_source_experiment.py \
  --versions v3 v4 \
  --attribution-mode all \
  --conditions consistent_easy \
  --max-items 1 \
  --prior-indices 0 \
  --skip-attention \
  --skip-layer-readout \
  --output-dir layer_metacognition/output/v3_v4_generation_only
```

中断后按原配置恢复：

```bash
python layer_metacognition/run_v3_v4_source_experiment.py \
  --dataset datasets/dataset_with_images.json \
  --image-root datasets \
  --model-path qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct \
  --output-dir layer_metacognition/output/v3_v4_all \
  --versions v3 v4 \
  --attribution-mode all \
  --conditions consistent_easy \
  --max-items 1 \
  --prior-indices 0 \
  --resume
```

恢复时，数据、attribution mode、analysis mode、condition、过滤条件和分析开关等配置必须与该目录的 `config.json` 一致。旧配置没有 `analysis_modes` 时会明确解释为旧默认 `["LMhead"]`；需要运行 Identity 或 Semantic 时必须使用新的输出目录。

## 输出文件

```text
<output-dir>/
├── config.json
├── progress.json
├── results.jsonl
├── analysis_minimal.json
├── analysis_layer_readout_minimal_v3.json
├── analysis_layer_readout_minimal_v4.json
├── analysis_source_sink_minimal.json
├── summary.json
└── run.log
```

- `results.jsonl`：每个 case 一条终态记录，成功和失败都会写入。
- `progress.json`：当前完成、失败和最后 case 信息。
- `analysis_minimal.json`：兼容旧四字段逐层结果。
- `analysis_layer_readout_minimal_v3.json`：只保存 V3 的 `case_id`、ground truths、`text_answer` 和七字段逐层 readout。
- `analysis_layer_readout_minimal_v4.json`：只保存 V4 的 `case_id`、ground truths 和七字段逐层 readout。
- `analysis_source_sink_minimal.json`：七字段逐层结果和逐 head sink。
- `summary.json`：成功/失败数量、逐模式 readout 覆盖、重构验证及 sink shape 汇总。
- `run.log`：运行日志。

三个分析 JSON 中的 case 都按版本分组保存：全部 V3 记录在前，全部 V4 记录在后，不会交叉；每个版本内部保持原 case 顺序以及 `none → parallel → joint` 的 mode 顺序。`results.jsonl` 为保证 append、fsync 和断点恢复安全，仍按实际完成顺序追加。

保存顺序为：

```text
每个 mode 结束
  → results.jsonl append + flush + fsync
  → progress.json 原子更新

同一 case 的全部 mode 结束
  → analysis_minimal.json 原子刷新
  → analysis_source_sink_minimal.json 原子刷新
  → summary.json 原子刷新
  → 进入下一个 case
```

`analysis_source_sink_minimal.json` 与两个 `analysis_layer_readout_minimal_<version>.json` 每层保存：

```text
[answer, answer_probability, answer_entropy,
 soft_confidence,
 LMhead_soft_image_score,
 Identity_soft_image_score,
 Semantic_soft_image_score]
```

未选择的 analysis mode 写 `null`。旧 `results.jsonl` 只有 `direct_readout["sac_layers"]` 时，该字段按 LMhead 解释，Identity/Semantic 自动写 `null`。`analysis_minimal.json` 继续保持原有四字段格式。

V3 每条记录示例：

```json
{
  "case_id": "1__prior_0__null__v3__none",
  "ground_truths": {
    "answer": "orange",
    "conflict_answer": "purple"
  },
  "text_answer": "orange",
  "layers": {
    "0": ["orange", 0.171, 2.280, 0.371, 0.514, null, null]
  }
}
```

所有版本都保存 `ground_truths`（原答案与 conflict answer）；只有 V3 保存 `text_answer`。

缺失的 readout 写为 `null`，不会删除该层。attention sink 使用 `target → source → layer → head` 索引，head 顺序保持模型原始 index。每个 sink layer 保存为：

```json
"0": {
  "sink_score_by_head": [0.00000312, 0.00000069],
  "attention_mass_by_head": [0.00381755, 0.00084830]
}
```

`sink_score_by_head` 是单位 source token 的平均注意力密度；比较图片和文本来源整体注意力时，应使用 `attention_mass_by_head`。

## 独立重新生成分析文件

```bash
python layer_metacognition/analyze_source_sink_results.py \
  --results layer_metacognition/output/v3_v4_all/results.jsonl \
  --output layer_metacognition/output/v3_v4_all/analysis_source_sink_minimal.json \
  --summary-output layer_metacognition/output/v3_v4_all/summary.json
```

## 测试

CPU 单元测试：

```bash
python -m unittest -v layer_metacognition.tests.test_v3_v4_source
```

静态语法检查：

```bash
python -m py_compile \
  layer_metacognition/run_v3_v4_source_experiment.py \
  layer_metacognition/v3_v4_source_runner.py
```

真实模型单 item smoke：

```bash
python layer_metacognition/smoke_test_v3_v4_source.py \
  --version v4 \
  --attribution-mode all \
  --output-dir layer_metacognition/output/smoke_v3_v4_source
```

smoke 的 `--attribution-mode` 同样支持 `none`、`parallel`、`joint` 和 `all`。使用 `all` 时会验证三条 mode 记录。完整逐层和逐 head 计算需要 CUDA 及足够显存；没有 CUDA 时仍可运行 CPU 单元测试和静态检查。

三种 SAC readout 的最小真实模型测试：

```bash
python layer_metacognition/smoke_test_v3_v4_source.py \
  --version v4 \
  --attribution-mode parallel \
  --analysis_mode LMhead Identity Semantic \
  --condition consistent_easy \
  --skip-attention \
  --output-dir layer_metacognition/output/sac_analysis_mode_smoke
```

smoke 会验证九类概率、score 范围、七列 JSON、逐模式最终层重构和 baseline，并打印所有 decoder layer 的 LMhead/Identity/Semantic `soft_image_score` 对照表。

三种方法的最后一层重构都会检查全词表 logits、restricted label、类别概率和 soft score。BF16 的全词表最大绝对误差容差为 `0.1`，用于容纳实际观察到的 `0.03125/0.0625` 量化步长；FP16 容差保持 `1e-3`。

## 旧 Layer Metacognition 入口

旧 AC/PANL/CC 主实验仍使用：

```bash
python layer_metacognition/run_main_experiment.py \
  --model-path qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct \
  --dataset datasets/dataset_with_images.json \
  --image-dir datasets \
  --layers all \
  --output-dir layer_metacognition/output/main \
  --resume
```

旧入口和新 V3/V4 Source Attribution 入口彼此独立。
