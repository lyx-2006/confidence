# Layer Metacognition 参数列表

## `run_main_experiment.py`

| 参数 | 类型 | 默认值 | 可选值或格式 | 作用 |
| --- | --- | --- | --- | --- |
| `-h`, `--help` | 开关 | 关闭 | 无参数值 | 显示 CLI 帮助并退出。 |
| `--model-path` | 路径 | `qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct` | 本地模型目录 | 指定 Qwen2.5-VL 模型目录。 |
| `--dataset` | 路径 | `datasets/datasets.json` | 数据集 JSON 文件 | 指定主实验数据集。 |
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
| `--dataset` | 路径 | `datasets/datasets.json` | 数据集 JSON 文件 | 指定评测数据集。 |
| `--image-root` | 路径 | 不设置 | 图片根目录 | 重定位数据集中的相对图片路径；不设置时使用数据集原有路径解析方式。 |
| `--model-path` | 路径 | `qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct` | 本地模型目录 | 指定 Qwen2.5-VL 模型目录。 |
| `--inference-path` | 路径 | `qwen-2.5-vl/inference.py` | Python 文件 | 指定 `QwenVLInference` 实现文件。 |
| `--output-dir` | 路径 | `layer_metacognition/output/v3_v4_source` | 输出根目录 | 每个 Source prompt variant 在该目录下使用独立子目录保存配置、进度、详细结果、分析结果和 hidden states；format version 1 的旧目录不能用于恢复。 |
| `--versions` | 字符串列表 | `v3 v4` | `v3`、`v4` | 选择运行版本，可指定一个或两个；多个值以空格分隔。 |
| `--attribution-mode` | 字符串 | `none` | `none`、`parallel`、`joint`、`all` | 选择 Source Attribution 生成方式；`all` 按 `none → parallel → joint` 全部运行。 |
| `--source-prompt-variant` | 字符串列表 | `baseline` | `baseline`、`answer_basis_9`、`answer_basis_10` | 选择一个或多个 joint SA prompt；对每个相同的 item × prior × condition × V3/V4 case，按 `baseline → answer_basis_9 → answer_basis_10` 顺序执行后再进入下一 case，产物分别写入 `<output-dir>/<variant>/`。两个 `answer_basis` 版本仅允许与单独的 `--attribution-mode joint` 一起使用。 |
| `--analysis_mode` | 字符串列表 | `LMhead` | `LMhead`、`Identity`、`Semantic` | 选择 SAC hidden-state readout；可指定多个，执行顺序固定为 `LMhead → Identity → Semantic`。Semantic Answer Patchscope 不受该参数控制，只要未跳过 layer readout 就会运行。 |
| `--conditions` | 字符串列表 | `all` | `null`、`irr`、`consistent_easy`、`consistent_hard`、`conflict_easy`、`conflict_hard`、`all` | 选择图片条件；可指定一个或多个。 |
| `--max-items` | 正整数 | 不限制 | `N ≥ 1` | 在 prior 和 condition 展开前限制原始数据 item 数量。 |
| `--item-ids` | 字符串列表 | 全部 | 一个或多个 item ID | 只运行指定 item；多个值可用空格或逗号分隔。 |
| `--prior-indices` | 整数列表 | 全部 | 一个或多个 `N ≥ 0` | 只运行指定的 text-prior 下标。 |
| `--resume` | 开关 | 关闭 | 无参数值 | 从同为 format version 2 的输出目录恢复，跳过已有 `case_id`。 |
| `--skip-attention` | 开关 | 关闭 | 无参数值 | 跳过逐 head attention sink，降低显存和运行时间，不影响逐层六字段 readout。 |
| `--skip-layer-readout` | 开关 | 关闭 | 无参数值 | 跳过 AC、CC、SAC、Semantic Answer Patchscope 及其最终层重构验证。 |
| `--save_probtable` | 开关 | 关闭 | 无参数值 | 将每个 case、每个 decoder layer 的 restricted Answer、Answer Patchscope、Confidence 和 SAC 类别概率另存为 variant 目录下的 `probability_tables.json`。 |
| `--skip_confidence` | 开关 | 关闭 | 无参数值 | 跳过最终 confidence 生成、CC teacher stage、逐层 CC readout 和 CC attention；V3 为 joint answer prompt 生成的 initial confidence 保留。 |
| `--answer_val` | 开关 | 关闭 | 无参数值 | 启用 Answer Patchscope 标签顺序验证：运行原始顺序和三个固定乱序版本；逐层 JSON 在原七列后追加三个乱序版本的 answer/prob，成为十三列；`summary.json["answer_validation"]` 保存对齐类别概率之间的逐层及总体 MAE、Pearson 相关系数。 |
| `--save_hidden_state` | 字符串或整数列表 | `none` | `none` 或一个/多个 0-based decoder layer index | 保存指定层、指定位置的 final norm 前 decoder-block output，例如 `--save_hidden_state 20 23 26`；层号按升序保存为 CPU FP16 分片。不能与 `--skip-layer-readout` 同时使用。 |
| `--save_hidden_state_positions` | 字符串列表 | `ac panl` | `ac panl ltt ptnl sac` | 选择 hidden-state token 位置；重复值去重并按固定顺序保存。显式指定位置时必须同时选择至少一个 layer。 |

当 tokenizer 将 Text clue 末尾字符与后续双换行融合为同一 token 时，PTNL
保留该融合 token，LTT 调整为 PTNL 前一个 processed token；结果中的
`position_adjustment` 会记录原始融合 token 和调整前后位置。
| `--max-answer-tokens` | 正整数 | `24` | `N ≥ 1` | 限制 answer 阶段最大生成 token 数。 |
| `--max-confidence-tokens` | 正整数 | `12` | `N ≥ 1` | 限制 confidence 阶段最大生成 token 数。 |
| `--max-source-tokens` | 正整数 | `4` | `N ≥ 1` | 限制 parallel Source Attribution 阶段最大生成 token 数。 |
