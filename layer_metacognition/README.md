# Layer Metacognition 参数列表

## `run_main_experiment.py`

| 参数 | 类型 | 默认值 | 可选值或格式 | 作用 |
| --- | --- | --- | --- | --- |
| `-h`, `--help` | 开关 | 关闭 | 无参数值 | 显示 CLI 帮助并退出。 |
| `--model-path` | 路径 | `qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct` | 本地模型目录 | 指定 Qwen2.5-VL 模型目录。 |
| `--dataset` | 路径 | `datasets/dataset_with_images.json` | 数据集 JSON 文件 | 指定主实验数据集。 |
| `--image-dir` | 路径 | 不设置 | 图片根目录 | 重定位数据集中的相对图片路径；不设置时以数据集所在目录为根目录。 |
| `--layers` | 字符串 | `all` | `all` 或逗号分隔的 decoder layer index | 选择执行 AC、PANL、CC readout 和保存 PANL hidden state 的层。 |
| `--save-hidden-states` | 字符串 | `panl` | `panl` | 指定保存 PANL hidden state；当前只有该选项。 |
| `--output-dir` | 路径 | `layer_metacognition/output/main` | 输出目录 | 保存结果、失败记录、元数据、分析文件和 hidden-state shards。 |
| `--max-items` | 正整数 | 不限制 | `N ≥ 1` | 在 prior 和图片条件展开前限制原始数据 item 数量。 |
| `--case-id` | 字符串 | 不设置 | 完整 case ID | 只运行指定 case，例如由 item、prior index 和图片条件组成的稳定 ID。 |
| `--shard-size` | 正整数 | `16` | `N ≥ 1` | 设置每个 PANL hidden-state shard 保存的 case 数量。 |
| `--resume` | 开关 | 关闭 | 无参数值 | 从已有输出恢复，修复尾部未完整 JSONL，并跳过已完成 case。 |
| `--retry-failures` | 开关 | 关闭 | 无参数值 | 恢复时重新运行 `failures.jsonl` 中已有的失败 case。 |

## `run_v3_v4_source_experiment.py`

| 参数 | 类型 | 默认值 | 可选值或格式 | 作用 |
| --- | --- | --- | --- | --- |
| `-h`, `--help` | 开关 | 关闭 | 无参数值 | 显示 CLI 帮助并退出。 |
| `--dataset` | 路径 | `datasets/dataset_with_images.json` | 数据集 JSON 文件 | 指定评测数据集。 |
| `--image-root` | 路径 | 不设置 | 图片根目录 | 重定位数据集中的相对图片路径；不设置时使用数据集原有路径解析方式。 |
| `--model-path` | 路径 | `qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct` | 本地模型目录 | 指定 Qwen2.5-VL 模型目录。 |
| `--inference-path` | 路径 | `qwen-2.5-vl/inference.py` | Python 文件 | 指定 `QwenVLInference` 实现文件。 |
| `--output-dir` | 路径 | `layer_metacognition/output/v3_v4_source` | 输出目录 | 保存配置、进度、详细结果和分析结果；format version 1 的旧目录不能用于恢复。 |
| `--versions` | 字符串列表 | `v3 v4` | `v3`、`v4` | 选择运行版本，可指定一个或两个；多个值以空格分隔。 |
| `--attribution-mode` | 字符串 | `none` | `none`、`parallel`、`joint`、`all` | 选择 Source Attribution 生成方式；`all` 按 `none → parallel → joint` 全部运行。 |
| `--analysis_mode` | 字符串列表 | `LMhead` | `LMhead`、`Identity`、`Semantic` | 选择 SAC hidden-state readout；可指定多个，执行顺序固定为 `LMhead → Identity → Semantic`。Semantic Answer Patchscope 不受该参数控制，只要未跳过 layer readout 就会运行。 |
| `--conditions` | 字符串列表 | `all` | `null`、`irr`、`consistent_easy`、`consistent_hard`、`conflict_easy`、`conflict_hard`、`all` | 选择图片条件；可指定一个或多个。 |
| `--max-items` | 正整数 | 不限制 | `N ≥ 1` | 在 prior 和 condition 展开前限制原始数据 item 数量。 |
| `--item-ids` | 字符串列表 | 全部 | 一个或多个 item ID | 只运行指定 item；多个值可用空格或逗号分隔。 |
| `--prior-indices` | 整数列表 | 全部 | 一个或多个 `N ≥ 0` | 只运行指定的 text-prior 下标。 |
| `--resume` | 开关 | 关闭 | 无参数值 | 从同为 format version 2 的输出目录恢复，跳过已有 `case_id`。 |
| `--skip-attention` | 开关 | 关闭 | 无参数值 | 跳过逐 head attention sink，降低显存和运行时间，不影响逐层六字段 readout。 |
| `--skip-layer-readout` | 开关 | 关闭 | 无参数值 | 跳过 AC、CC、SAC、Semantic Answer Patchscope 及其最终层重构验证。 |
| `--answer_val` | 开关 | 关闭 | 无参数值 | 启用 Answer Patchscope 标签顺序验证：运行原始顺序和三个固定乱序版本；逐层 JSON 在原七列后追加三个乱序版本的 answer/prob，成为十三列；`summary.json["answer_validation"]` 保存对齐类别概率之间的逐层及总体 MAE、Pearson 相关系数。 |
| `--save_hidden_state` | 字符串或整数列表 | `none` | `none` 或一个/多个 0-based decoder layer index | 保存指定层 AC 与 PANL token 的 final norm 前 decoder-block output，例如 `--save_hidden_state 20 23 26`；层号按升序保存为 CPU FP16 分片。不能与 `--skip-layer-readout` 同时使用。 |
| `--max-answer-tokens` | 正整数 | `24` | `N ≥ 1` | 限制 answer 阶段最大生成 token 数。 |
| `--max-confidence-tokens` | 正整数 | `12` | `N ≥ 1` | 限制 confidence 阶段最大生成 token 数。 |
| `--max-source-tokens` | 正整数 | `4` | `N ≥ 1` | 限制 parallel Source Attribution 阶段最大生成 token 数。 |
