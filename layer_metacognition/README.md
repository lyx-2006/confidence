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

## Phase 2：Decision-Side Steering

### `run_decision_side_steering.py` 参数表

| 参数 | 类型 | 默认值 | 可选值或格式 | 作用 |
| --- | --- | --- | --- | --- |
| `-h`, `--help` | 开关 | 关闭 | 无参数值 | 显示参数帮助并退出。 |
| `--experiment-dir` | 路径 | `output/Final_v4_run/answer_basis_9` | 已完成的 V4 experiment 目录 | 读取原始 `results.jsonl`，作为 alpha=0 的 Steering 前 baseline。 |
| `--probe-run-dir` | 路径 | 不设置 | `stage1_metacognition/item_split` 目录 | 读取 Stage 1 `run_config.json`、item fold、direction index 和 direction NPZ。 |
| `--layers` | 整数列表 | 必填 | 一个或多个非负 layer index | 指定运行层；只运行明确列出的层。 |
| `--positions` | 字符串列表 | 必填 | `ptnl`、`ac`、`panl` | 指定 Steering 位置。PANL 是 post-answer 位置，只在 teacher-forced pass 注入，因此不用于检验 Answer effect。 |
| `--alphas` | 浮点列表 | 必填 | 有限浮点数，必须恰好包含一个 `0` | 指定 Steering 强度；`+alpha` 指向 `follows_image`。 |
| `--steering-scale` | 字符串 | `probe_logit` | `probe_logit`、`unit` | 选择 Steering 向量缩放方式；正式实验使用 `probe_logit`。 |
| `--injection-site` | 字符串 | `block_output` | `block_output`、`block_input` | `block_output` 是原实验定义；`block_input` 在目标 block 计算 attention/KV 前修改 residual，作为同层 K/V 可见性对照。后者使用 post-block Probe direction，因此必须按诊断对照解释。 |
| `--intervention-mode` | 字符串 | `single` | `single`、`reinject` | `single` 只在指定层注入；`reinject` 从指定层起，在每个存在 OOF direction 的后续层用各层自己的 direction 再注入同一 alpha。`reinject` 仅支持 `block_output`。 |
| `--conditions` | 字符串列表 | `conflict_easy conflict_hard` | 仅允许 `conflict_easy`、`conflict_hard` | 选择 conflict case；不会混入 consistent、irr 或 null case。 |
| `--cases-per-decision-side` | 正整数 | `150` | `N ≥ 1` | 在其他筛选完成后，按稳定 case 顺序分别选取 `N` 条 `follows_text` 和 `N` 条 `follows_image`。默认共 300 条。 |
| `--item-ids` | 字符串列表 | 全部 | 一个或多个 item ID | 只保留指定 item；应用后仍需满足两类各 `--cases-per-decision-side` 条。 |
| `--prior-indices` | 整数列表 | 全部 | 一个或多个 `N ≥ 0` | 只保留指定 prior index；应用后仍需满足两类样本数。 |
| `--max-cases` | 正整数 | 不限制 | `N ≥ 1` | 在平衡抽样完成后再截断 case 数；正式 150+150 运行不设置此参数。 |
| `--max-baseline-abs-answer-margin` | 正浮点数 | 不限制 | `x > 0` | 在平衡抽样前只保留 Steering 前 `abs(AnswerMargin) < x` 的近边界 case；使用后需同步降低 `--cases-per-decision-side`，以免某一类样本不足。 |
| `--model-path` | 路径 | `qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct` | 本地 Qwen 模型目录 | 指定 Steering 使用的模型。必须完整放在 CUDA，禁止 CPU/disk offload。 |
| `--dataset` | 路径 | `datasets/datasets.json` | 数据集 JSON 文件 | 根据 manifest case 还原问题、候选答案和图片路径。 |
| `--image-root` | 路径 | 不设置 | 图片根目录 | 可选地重定位数据集中的相对图片路径。 |
| `--inference-path` | 路径 | `qwen-2.5-vl/inference.py` | Python 文件 | 指定 Qwen inference loader。 |
| `--output-dir` | 路径 | 不设置 | 不存在的新目录 | 保存 `run_config.json`、`results.jsonl`、`progress.json` 和 `summary.json`。 |
| `--max-answer-tokens` | 正整数 | `24` | `N ≥ 1` | 限制 answer generation 的最大新 token 数。 |
| `--resume` | 开关 | 关闭 | 无参数值 | 使用完全相同配置恢复；配置、manifest 或 direction fingerprint 不一致时拒绝。 |

alpha=0 的 Answer/SA 只从 experiment 的原始 `results.jsonl` 复制，不执行 Qwen
generation、teacher-forced forward 或 activation hook；逐层 K baseline 从原始
hidden-state shards 读取。非零 alpha 才执行模型推理；单条失败会写入
`results.jsonl` 并继续后续任务。默认的 300 条数据分布为 `follows_text=150`、
`follows_image=150`。

`run_decision_side_steering.py` 使用 Phase 1 item-split Probe 为每个 case 选择其
自身 OOF fold 的 direction，并在 V4 `answer_basis_9` joint prompt 的 PTNL、AC 或 PANL
位置执行 additive steering。`+alpha` 固定指向 `follows_image`；不会平均 fold
direction，也不会重训 Probe。

正式 scale `probe_logit` 定义为：

```text
h' = h + alpha * d_raw / ||d_raw||^2
```

因此 raw Probe logit 的理论变化为 `alpha`。可选 `unit` scale 使用
`h' = h + alpha*d_K`。生成阶段的 hook 只在完整 prompt prefill 上注入一次；
`use_cache=True` 的后续单 token decode 不再注入。teacher-forced SAC scoring 会用
steered generated answer 重建相同 joint wire format，再在同一 layer、position、
alpha 和 OOF direction 下重新注入并调用现有 SA scorer。

默认 `single + block_output` 与此前结果完全同义。`single + block_input` 把同一
direction 加到目标 block 的输入 residual，使该 block 自身的 Q/K/V 投影可以看到
扰动；它回答的是“缓存可见性是否限制了单层 output Steering”，但由于 OOF Probe
是在 block output 上训练的，不能把 input-site manipulation check 当成 output Probe
已经移动。`reinject + block_output` 则检查模型在后续层持续被拉回 Decision axis 时
Answer/SA 是否才发生变化。每层实际注入检查分别写入
`generation_reinjection_diagnostics` 和 `teacher_forced_reinjection_diagnostics`；
其 trajectory 语义为 `cumulative_reinjection_response`，不得与单次注入的
retention 曲线混合。

PANL 是答案字段之后、Source Attribution 字段之前的 newline token。它在生成答案时
尚不存在，所以 PANL generation 不安装 hook；代码先生成未干预答案，再在 joint
teacher-forced pass 的 PANL 注入。PANL 的 `answer_intervention_applicable=false`，
只解释其对 SA 和后续逐层表示的影响。

每条结果的 `layer_trajectory` 按注入层及其后所有已有 OOF Decision-Side Probe 的层
保存：`baseline_logit`、`steered_logit`、`delta_logit`、`baseline_K`、`steered_K`、
`delta_K` 和 `retention_fraction`。每个 readout layer 使用自己的 item-split OOF
direction。当前可用层为 `12 14 16 18 20 22 24 26 27`；例如 L22 注入后严格读取
L22/L24/L26/L27，不使用不存在的 L23 Probe。

trajectory 还会保存完整 hidden-state 扰动的几何诊断：`delta_hidden_l2`、
`delta_hidden_projection_on_d_K`、`delta_hidden_cosine_with_d_K`、
`delta_hidden_orthogonal_l2` 和 `directional_energy_fraction`。这可以区分两种情况：
完整扰动本身快速衰减，或完整扰动仍然存在但被旋转到后续层 Decision Probe 的
正交子空间。注入层使用同一次 forward 的 `h_before` 作为向量基线；后续层使用原始
实验 hidden states，并用 `delta_hidden_baseline_source` 明确标记。若 Steering 改变了
generated answer，teacher-forced wire 与保存的 baseline 不再相同，
`answer_context_matches_saved_baseline=false`，分析传播时应单独剔除或报告这些记录。

```text
retention_fraction(readout_layer)
  = delta_logit(readout_layer) / in_run_delta_logit(injection_layer)
```

分母来自同一次 forward 中注入层实际测得的 before/after logit shift；注入层 retention
固定为 1。`saved_baseline_alignment_logit_error` 记录原始 hidden-state baseline 与当前
forward 注入前读数的差异。`summary.json["trajectory_cells"]` 按 injection layer ×
readout layer × position × alpha × subgroup 汇总 delta logit、delta K 和 retention。
同时汇总上述 hidden 几何量，并额外生成
`baseline_abs_answer_margin_lt_0.5`、`baseline_abs_answer_margin_lt_1`、
`baseline_abs_answer_margin_ge_1` 三组，用于单独检查靠近答案决策边界的 case。

Manipulation check 分开记录请求向量的理论 delta 与低精度 activation 中的实际
delta：请求向量以 `1e-8` 容差验证公式；BF16 写回使用 25%（至少 0.25）的显式
量化容差，同时仍要求实际 delta 方向正确。FP16/FP32 保持 10%（至少 0.1）容差。

程序只允许 CUDA Qwen 推理。CUDA 不可用或 device map 含 CPU/disk offload 时会在
实验开始前失败，不存在 CPU fallback。alpha=0 不再运行 Qwen forward，而是直接
复用 Steering 前 `results.jsonl` 中的 answer probabilities、AnswerMargin 和 soft SA，
并为每个 layer × position 写入 baseline record 供 paired analysis 使用。默认在所有
筛选后按稳定 case 顺序分别选择 150 条 `follows_text` 和 150 条 `follows_image`；
可用 `--cases-per-decision-side` 调整。其他单条 intervention 失败会保存为 failed
record 后继续运行。

20-case smoke test 使用独立目录，避免污染正式输出：

```bash
cd /root/autodl-tmp
python -u -m layer_metacognition.run_decision_side_steering \
  --experiment-dir layer_metacognition/output/Final_v4_run/answer_basis_9 \
  --probe-run-dir layer_metacognition/output/Final_v4_run/answer_basis_9/stage1_metacognition/item_split \
  --layers 22 \
  --positions ac \
  --alphas -2 -1 0 1 2 \
  --steering-scale probe_logit \
  --conditions conflict_easy conflict_hard \
  --max-cases 20 \
  --output-dir layer_metacognition/output/Final_v4_run/answer_basis_9/stage2_decision_steering_smoke
```

同时检查 AC/PANL 跨层 retention 的诊断 smoke：

```bash
cd /root/autodl-tmp
python -u -m layer_metacognition.run_decision_side_steering \
  --experiment-dir layer_metacognition/output/Final_v4_run/answer_basis_9 \
  --probe-run-dir layer_metacognition/output/Final_v4_run/answer_basis_9/stage1_metacognition/item_split \
  --layers 22 \
  --positions ac panl \
  --alphas -5 0 5 \
  --steering-scale probe_logit \
  --conditions conflict_easy conflict_hard \
  --cases-per-decision-side 150 \
  --max-cases 20 \
  --output-dir layer_metacognition/output/Final_v4_run/answer_basis_9/stage2_decision_trajectory_smoke
```

建议用三个互不复用的输出目录比较单层 output、同层 K/V 可见 input 和逐层重注入。
后两项的最小 AC smoke 分别为：

```bash
cd /root/autodl-tmp
python -u -m layer_metacognition.run_decision_side_steering \
  --experiment-dir layer_metacognition/output/Final_v4_run/answer_basis_9 \
  --probe-run-dir layer_metacognition/output/Final_v4_run/answer_basis_9/stage1_metacognition/item_split \
  --layers 22 --positions ac --alphas -5 0 5 \
  --steering-scale probe_logit --injection-site block_input \
  --intervention-mode single --max-cases 20 \
  --output-dir layer_metacognition/output/Final_v4_run/answer_basis_9/stage2_ac_block_input_smoke

python -u -m layer_metacognition.run_decision_side_steering \
  --experiment-dir layer_metacognition/output/Final_v4_run/answer_basis_9 \
  --probe-run-dir layer_metacognition/output/Final_v4_run/answer_basis_9/stage1_metacognition/item_split \
  --layers 22 --positions ac --alphas -5 0 5 \
  --steering-scale probe_logit --injection-site block_output \
  --intervention-mode reinject --max-cases 20 \
  --output-dir layer_metacognition/output/Final_v4_run/answer_basis_9/stage2_ac_reinject_smoke
```

近边界诊断可追加 `--max-baseline-abs-answer-margin 1`，并将
`--cases-per-decision-side` 调整到筛选后两类都能满足的数量。当前 1300 条 eligible
数据中，`|AnswerMargin|<1` 为 follows_text 59 条、follows_image 55 条，因此可用
`--cases-per-decision-side 55`；`|AnswerMargin|<0.5` 则分别为 35/27，最多平衡取 27。

smoke test 全部通过后，建议的全量命令为（本程序不会自动启动它）：

```bash
cd /root/autodl-tmp
python -u -m layer_metacognition.run_decision_side_steering \
  --experiment-dir layer_metacognition/output/Final_v4_run/answer_basis_9 \
  --probe-run-dir layer_metacognition/output/Final_v4_run/answer_basis_9/stage1_metacognition/item_split \
  --layers 12 14 16 18 20 22 24 26 27 \
  --positions ptnl ac \
  --alphas -2 -1 0 1 2 \
  --steering-scale probe_logit \
  --conditions conflict_easy conflict_hard \
  --cases-per-decision-side 150 \
  --output-dir layer_metacognition/output/Final_v4_run/answer_basis_9/stage2_decision_steering
```

中断恢复必须使用完全相同的参数并追加 `--resume`；配置或 Stage 1 artifact
fingerprint 不一致时拒绝恢复。输出包括 `run_config.json`、`results.jsonl`、
`progress.json` 和 `summary.json`。`summary.json` 仅使用同 case、layer、position 的
alpha=0 配对基线计算 `Delta AnswerMargin` 与 `Delta SA`。Steering 是对所选 Probe
direction 的因果干预，但单次结果仍不等同于对所有 arbitration mechanism 的完整
识别。

## Teacher-Forced Source Attribution causal experiments

独立入口固定使用 V4 `answer_basis_9` prompt，并以相同 forced Answer 的 clean
Teacher-Forced SA 作为每次 replacement 的 causal baseline：

```bash
cd /root/autodl-tmp
python -m layer_metacognition.run_teacher_forced_source_origin --smoke
```

去掉 `--smoke` 才会在默认目录 `stage2_teacher_forced_source_origin` 构建正式的
100-case cohort；程序不会由 smoke 自动扩展到正式运行。默认干预层为
L12/16/20/24/26，而 clean capture 始终覆盖全部 decoder layers。大 image/text
state 在 Evidence Swap 后删除，小 AC/PANL/SAC state 在 State Swap 后删除；
`precomputed_states/index.json` 保留各 context 的 layer、shape、dtype、位置和删除记录。
中断恢复需使用完全相同的参数并追加 `--resume`。

## Strong-SA mean-difference steering

`run_sa_mean_steering` 从已完成的 V4 conflict baselines 中按
`SA_soft_image_score` 选择 follows_image 最强 25 条与 follows_text 最强 25 条。
每组只保留一个 case/item，两组不复用 item；evaluation 也与这 50 个 source item
完全隔离。每个已保存 layer × AC/PANL 分别构造
`mean(image) - mean(text)`，实际注入量为 `alpha * mean_difference`。正 alpha 为
imageward，负 alpha 为 textward。

最小 smoke：

```bash
cd /root/autodl-tmp
python -u -m layer_metacognition.run_sa_mean_steering \
  --layers 20 --positions ac panl --alphas -1 0 1 \
  --max-cases 4 \
  --output-dir layer_metacognition/output/Final_v4_run/answer_basis_9/stage2_sa_mean_steering_smoke
```

默认正式网格为 L20/L24 × AC/PANL × alpha `-1 -0.5 0 0.5 1`，held-out
evaluation 为 follows_text/image 各 25 条。输出额外包含
`source_cohort_manifest.json`、`evaluation_manifest.json` 和 `directions/`；中断后用
完全相同参数追加 `--resume`。
