# V3/V4 Joint Hidden-State Linear Probe

本包在已有 joint inference 的 hidden-state shard 上训练逐层线性 Probe，用于测量
模型内部是否存在线性可解码的单模态答案与冲突信息。

当前版本提供两套后端：

- `torch`：固定 C、支持 CUDA，适合快速扫描 layer 和 token 位置。
- `sklearn`：支持 grouped inner-C 和 permutation baseline，是正式复验后端。

Probe 不会重新运行 joint inference，也不会改写 `results.jsonl`、hidden-state
index 或 shard。标签生成是唯一可能加载 Qwen 模型的步骤；训练 Probe 本身不
加载 Qwen。

> Linear Probe 只说明信息可以被线性解码，不足以证明这些信息被模型因果使用。

## Phase 1：Decision-Side / Arbitration Probe

Decision-Side target 只从模型实际行为生成。对 `conflict_easy` / `conflict_hard`
case，当 text-only 与 image-only answer 都有效且不同时：joint current answer 等于
text-only answer 记为 `follows_text`，等于 image-only answer 记为
`follows_image`。`follows_neither`、相同 unimodal answer 和缺失 label 均不参与。
Manifest 保留字符串 label；训练时固定 `follows_text -> 0`、
`follows_image -> 1`，所以 `P(class1)` 和 `+d_K` 始终指向 image-side。

训练入口新增：

```text
--decision-side-probe-location ptnl ac panl
--split-mode item|answer_pair
--decision-side-only
```

不传 `--decision-side-probe-location` 时不创建 Decision-Side task。`item` 是默认
模式，原 tasks 与 Decision-Side tasks 复用既有 item-grouped folds。
`answer_pair` 模式只训练 Decision-Side：先按无序 answer pair 分 test fold，再从
train 清除 test items，同时保证 pair 和 item 均无泄漏。该模式使用独立的
`decision_side_pair_split_assignments.json`，不会复用或替换旧 item assignments。
已有 R_T/R_I/C OOF 时，item run 可显式传 `--decision-side-only`，只训练新增
Decision-Side tasks；该开关不改变未传时的旧默认行为。

每个 Decision-Side outer-fold test case 保存 OOF `P(follows_text)` 与
`P(follows_image)`。每个 fold × position × layer 还在 `decision_directions/` 保存
scaler、标准化 weight/intercept、原空间 `d_raw`、raw intercept 和单位向量
`d_K`。这些方向供后续阶段使用；本阶段不执行 steering。

### 正式 Phase 1 运行

```bash
cd /root/autodl-tmp

python -u -m layer_metacognition.probe.build_probe_manifest \
  --experiment-dir layer_metacognition/output/Final_v4_run/answer_basis_9 \
  --output-dir layer_metacognition/output/Final_v4_run/answer_basis_9/extended_probe

python -u -m layer_metacognition.probe.train_layer_probes \
  --experiment-dir layer_metacognition/output/Final_v4_run/answer_basis_9 \
  --output-dir layer_metacognition/output/Final_v4_run/answer_basis_9/stage1_metacognition/item_split \
  --manifest-path layer_metacognition/output/Final_v4_run/answer_basis_9/extended_probe/probe_manifest.jsonl \
  --layers 12 14 16 18 20 22 24 26 27 \
  --n-splits 5 --seed 42 \
  --probe-conditions consistent_easy consistent_hard conflict_easy conflict_hard \
  --decision-side-probe-location ptnl ac panl \
  --decision-side-only \
  --version-settings v4_to_v4 --split-mode item \
  --backend torch --fixed-c 1.0 --device cuda --permutations 0

python -u -m layer_metacognition.probe.train_layer_probes \
  --experiment-dir layer_metacognition/output/Final_v4_run/answer_basis_9 \
  --output-dir layer_metacognition/output/Final_v4_run/answer_basis_9/stage1_metacognition/answer_pair_split \
  --manifest-path layer_metacognition/output/Final_v4_run/answer_basis_9/extended_probe/probe_manifest.jsonl \
  --layers 12 14 16 18 20 22 24 26 27 \
  --n-splits 5 --seed 42 \
  --probe-conditions consistent_easy consistent_hard conflict_easy conflict_hard \
  --decision-side-probe-location ptnl ac panl \
  --version-settings v4_to_v4 --split-mode answer_pair \
  --backend torch --fixed-c 1.0 --device cuda --permutations 0
```

Trajectory 复用现有 text/image/conflict OOF 和 SAC Semantic readout：

```bash
python -u -m layer_metacognition.probe.analyze_stage1_trajectory \
  --experiment-dir layer_metacognition/output/Final_v4_run/answer_basis_9 \
  --manifest-path layer_metacognition/output/Final_v4_run/answer_basis_9/extended_probe/probe_manifest.jsonl \
  --probe-results-dir layer_metacognition/output/Final_v4_run/answer_basis_9/extended_probe_torch_manual \
  --decision-item-results-dir layer_metacognition/output/Final_v4_run/answer_basis_9/stage1_metacognition/item_split \
  --decision-answer-pair-results-dir layer_metacognition/output/Final_v4_run/answer_basis_9/stage1_metacognition/answer_pair_split \
  --output-dir layer_metacognition/output/Final_v4_run/answer_basis_9/stage1_metacognition \
  --layers 12 14 16 18 20 22 24 26 27 \
  --decision-side-probe-location ptnl ac panl
```

`R_I_preliminary` 只有 PTNL/AC/PANL 基础 readout。当前没有保存
post-image-token hidden state，因此它不能回答 image candidate 的完整形成时间，
也不能比较 text/image candidate 谁更早形成。PANL 是 answer 生成后的 token；
PANL 上高 K 不证明 pre-answer arbitration。Phase 1 只报告线性可解码性、预测、
跨 unseen answer-pair 泛化及相关关系，不作 `K causes Answer/SA` 等因果结论。

## 1. 目录与依赖

所有实现、测试与文档均位于：

```text
layer_metacognition/probe/
```

安装正式 sklearn Probe 和测试依赖：

```bash
cd /root/autodl-tmp
python -m pip install -r layer_metacognition/probe/requirements.txt
```

`requirements.txt` 声明：

```text
scikit-learn>=1.6,<2
pytest>=8
```

Torch 是可选依赖，采用 lazy import，不写入 Probe requirements。使用 GPU
后端前可检查：

```bash
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

## 2. 预测目标与位置

Probe 使用模型行为标签，不使用数据集 ground truth：

- `A_T / text_only_answer`：模型只读取 Question + Text clue 时的答案。
- `A_I / image_only_answer`：模型只读取 Question + Image 时的答案。
- `A_J / current_answer`：现有 joint run 的 `generated.current_answer`。
- `conflict_label`：由 condition 映射为 `consistent` 或 `conflict`。

所有答案同时保存 raw 字符串和 `normalize_answer` 规范化结果。数据集字段
`answer`、`text_ans`、`conflict_ans` 不作为 Probe 行为标签。

支持五个 hidden 位置：

| 位置 | 含义 |
| --- | --- |
| `ac` | 当前答案输出前的 answer cognition token |
| `panl` | 当前答案输出后的换行 token |
| `ltt` | Text clue 最后一个非空白 token |
| `ptnl` | Text clue 后双换行分隔符的最后一个 token |
| `sac` | Source Attribution marker 的冒号 token |

每个 Answer 位置产生两个任务：`<position>_text_answer` 和
`<position>_image_answer`；每个 Conflict 位置产生
`<position>_conflict`。默认位置是 `ac panl`。

每个 hidden Probe 只使用一个 layer、一个位置的二维输入
`[batch, hidden_size]`，不拼接 layer/position，也不把 layer index 当作特征。

## 3. 条件与分析子集

可选条件为：

```text
consistent_easy
consistent_hard
conflict_easy
conflict_hard
```

默认只选择：

```text
consistent_easy conflict_easy
```

训练使用所选条件的 pooled 数据；condition 不单独调参。每个有效 fold 都输出：

- `pooled_overall`
- `easy_overall`
- 四个单独 condition
- `discriminative_conflict`
- `joint_follows_text`
- `joint_follows_image`
- `joint_follows_neither`

空子集仍会写入结果，状态为 `empty`，计数为 0，数值指标为 `null`。

### 当前 extended manifest 的注意事项

当前路径实际拼写为：

```text
layer_metacognition/output/entended_datasets/answer_basis_9
```

已有 manifest 位于：

```text
layer_metacognition/output/entended_datasets/answer_basis_9/extended_probe/probe_manifest.jsonl
```

该 manifest 有 3,260 条 V4 case。Text 和 Conflict 标签覆盖四种条件，但现有
Image 标签仅覆盖 1,619 条 easy case。因此直接训练时：

- Text Probe：可使用 easy + hard。
- Conflict Probe：可使用 easy + hard。
- Image Probe：只使用具有有效 `A_I` 的 easy case，hard subset 会是空的。

若研究目标需要 Image hard 指标，必须先按第 6 节补生成 hard image-only 标签，
再重建 manifest。这会加载 Qwen 模型，耗时与快速 Probe 训练无关。

## 4. 数据划分与有效性规则

Outer split 使用按 `item_id` 分组的 shuffled GroupKFold。默认：

```text
n_splits = 5
seed = 42
```

所有任务、位置、layer 和版本设置复用同一份 `split_assignments.json`。同一道题
的 prior、condition 和 V3/V4 case 始终位于同一个 outer fold。跨版本测试仍只
使用 outer test items，不会用同一 item 的另一个版本训练。

版本设置包括：

| 设置 | Train | Test |
| --- | --- | --- |
| `v3_to_v3` | V3 | V3 |
| `v4_to_v4` | V4 | V4 |
| `v3_to_v4` | V3 | V4 |
| `v4_to_v3` | V4 | V3 |

默认请求全部四种设置。V4-only manifest 必须显式指定：

```text
--version-settings v4_to_v4
```

以下情况会把完整 task/version/fold 标记为 invalid：

- outer train 或 test 为空；
- outer train 只有一个类别；
- outer test 包含训练阶段未见类别；
- Torch 拟合或预测出现 NaN/Inf。

不会删除单条测试记录，也不会使用 test label 扩展 LabelEncoder。hidden shard
损坏、offset/case ID 不一致等 integrity error 会终止整个运行。

## 5. 双后端

### 5.1 Torch 快速扫描后端

Torch 后端要求：

```text
--backend torch
--fixed-c <positive float>
--permutations 0
```

`--device` 可设为 `auto`、`cpu` 或 `cuda`。GPU 只保存当前拟合所需 tensor，
不会把全部 shard 常驻显存。

训练定义：

- scaler 只 fit outer-train 数据，并复用 sklearn `StandardScaler` 语义；
- balanced sample weights；
- Answer Probe：`K` logits + weighted cross-entropy；
- Conflict Probe：单 logit + weighted binary cross-entropy；
- L2 objective 与 sklearn 的 C 缩放对齐，intercept 不正则化；
- full-batch LBFGS，先最多 100 iterations，必要时继续到总计 200。

收敛条件为 final loss 有限，并满足 gradient infinity norm ≤ `1e-5` 或 relative
loss change ≤ `1e-7`。200 次后仍未收敛但数值有效时保留预测并标记
`converged=false`；NaN/Inf 将该 model/fold 标记为 invalid。

### 5.2 sklearn 正式后端

默认模型为：

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        penalty="l2",
        C=selected_C,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=5000,
    )),
])
```

未指定 `--fixed-c` 时，outer-train 内执行 3-fold grouped CV，从以下网格按
balanced accuracy 选择 C，平分取较小值：

```text
0.01 0.1 1.0 10.0
```

组数不足、inner train 单类别或 inner validation 出现未见类别时回退
`C=1.0` 并记录原因。

### 5.3 公平比较两个后端

后端 parity 必须使用相同固定 C，例如：

```text
sklearn --fixed-c 1.0
torch   --fixed-c 1.0
```

不能拿 Torch fixed-C 与 sklearn inner-C 结果比较，否则差异同时包含后端和 C
选择差异。

## 6. 数据准备：标签与 manifest

如果已有标签和 manifest，可直接跳到第 7 节。

### 6.1 生成或恢复默认 easy 标签

```bash
cd /root/autodl-tmp

python -u -m layer_metacognition.probe.generate_unimodal_labels \
  --experiment-dir layer_metacognition/output/Final_v4_run/answer_basis_9 \
  --output-dir layer_metacognition/output/Final_v4_run/answer_basis_9/extended_probe \
  --dataset datasets/datasets.json \
  --model-path qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct \
  --probe-conditions consistent_easy conflict_easy \
  --resume
```

脚本先从现有 V3 `generated.initial_answer` 聚合 Text 标签。同键多个规范化答案
冲突时立即失败。只有 Text 标签缺失或所选 Image 标签尚未成功时才加载模型；
model 和 processor 在进程内只加载一次。

`--resume` 跳过成功标签并重试失败标签。失败不会由 ground truth 补齐，会写入
`label_failures.jsonl`。

### 6.2 可选：补齐四条件 Image 标签

仅当确实需要 Image hard Probe 时运行：

```bash
python -u -m layer_metacognition.probe.generate_unimodal_labels \
  --experiment-dir layer_metacognition/output/Final_v4_run/answer_basis_9 \
  --output-dir layer_metacognition/output/entended_datasets/answer_basis_9/extended_probe \
  --dataset datasets/datasets.json \
  --model-path qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct \
  --probe-conditions consistent_easy consistent_hard conflict_easy conflict_hard \
  --resume
```

### 6.3 重建 manifest

标签变化后必须重建 manifest：

```bash
python -u -m layer_metacognition.probe.build_probe_manifest \
  --experiment-dir layer_metacognition/output/Final_v4_run/answer_basis_9 \
  --output-dir layer_metacognition/output/Final_v4_run/answer_basis_9/extended_probe
```

Manifest 只收录 completed joint case、有效 current answer 和有效 hidden reference。
缺失 Text/Image 标签不会被虚构，相应 eligibility 为 false。

`manifest_summary.json` 保存：

- 来源实验绝对路径；
- hidden-state index SHA-256；
- dataset fingerprint；
- manifest SHA-256；
- records/items/conditions/versions 与标签缺失统计。

## 7. 当前 V4 GPU 加速扫描：手动前台运行

以下命令直接复用现有 manifest，不会加载 Qwen，也不需要 screen/nohup。

### 7.1 设置环境与路径

```bash
cd /root/autodl-tmp

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1

export PROBE_EXP=layer_metacognition/output/Final_v4_run/answer_basis_9
export PROBE_MANIFEST=layer_metacognition/output/Final_v4_run/answer_basis_9/extended_probe/probe_manifest.jsonl
export PROBE_OUT=layer_metacognition/output/Final_v4_run/answer_basis_9/extended_probe_torch_manual

mkdir -p "$PROBE_OUT"
```

### 7.2 确认没有残留训练进程

```bash
pgrep -af layer_metacognition.probe.train_layer_probes || echo "没有残留进程"
```

### 7.3 启动 Torch CUDA 扫描

```bash
python -u -m layer_metacognition.probe.train_layer_probes \
  --experiment-dir "$PROBE_EXP" \
  --output-dir "$PROBE_OUT" \
  --manifest-path "$PROBE_MANIFEST" \
  --layers 12 14 16 18 20 22 24 26 27 \
  --n-splits 3 \
  --seed 42 \
  --probe-conditions consistent_easy consistent_hard conflict_easy conflict_hard \
  --answer-probe-location ac panl ltt ptnl sac \
  --conflict-probe-location ac panl ltt ptnl sac \
  --version-settings v4_to_v4 \
  --backend torch \
  --fixed-c 1.0 \
  --device cuda \
  --permutations 0
```

这次运行包含 15 个 task/location、7 个 layer、3 folds，共 315 次 hidden Probe
拟合；PANL 另外训练 6 次 current-answer baseline。运行时间是性能目标而不是
正确性门槛，以输出中的 timing 和 iteration 统计为准。

### 7.4 汇总结果

训练命令退出码为 0 后运行：

```bash
python -u -m layer_metacognition.probe.analyze_probe_results \
  --experiment-dir "$PROBE_EXP" \
  --output-dir "$PROBE_OUT"
```

主要结果：

```text
$PROBE_OUT/probe_summary.json
$PROBE_OUT/layer_probe_metrics.json
$PROBE_OUT/layer_probe_predictions.jsonl
```

## 8. sklearn 正式复验

GPU scan 用于选层和定位；正式报告使用 sklearn。下面是 V4-only 的缩减示例，
应根据快速扫描结果替换 layers/locations：

```bash
export FORMAL_OUT=layer_metacognition/output/Final_v4_run/answer_basis_9/extended_probe_sklearn_formal

python -u -m layer_metacognition.probe.train_layer_probes \
  --experiment-dir "$PROBE_EXP" \
  --output-dir "$FORMAL_OUT" \
  --manifest-path "$PROBE_MANIFEST" \
  --layers 20 27 \
  --n-splits 5 \
  --seed 42 \
  --probe-conditions consistent_easy consistent_hard conflict_easy conflict_hard \
  --answer-probe-location ac panl \
  --conflict-probe-location ac panl \
  --version-settings v4_to_v4 \
  --backend sklearn \
  --permutations 20
```

如果 manifest 同时包含 V3/V4，可显式运行四种设置：

```text
--version-settings v3_to_v3 v4_to_v4 v3_to_v4 v4_to_v3
```

### 对照模型

PANL Answer task 会按 task/fold/train-version 额外训练一次
`current_answer_only_baseline`：

```text
A_J/current_answer -> A_T 或 A_I
```

该 baseline 用于估计答案输出后位置的 current-answer leakage，不会逐 layer 重复
训练。

sklearn 默认运行 20 个 permutation seeds。标签只在 outer-train 的唯一行为键
上打乱，测试标签不变：

- Text：`(item_id, prior_index)`；
- Image/Conflict：`(item_id, condition)`。

Permutation model 复用真实 Probe 选出的 C，不重新执行 inner tuning。

## 9. 缓存与性能优化

新版训练器保持 manifest 原始行顺序，并执行两级复用：

- Feature cache：每个 `(experiment fingerprint, manifest fingerprint, position,
  layer)` 只构造一次完整 float32 matrix。
- Model cache：同一个 task/layer/fold/train-version 只拟合一次，再用于对应的同
  版本和跨版本 test setting。

模型缓存键还包含 target、backend、C、seed、conditions、permutation seed 和
数据 fingerprints。current-answer baseline、C selection 和 permutation model
同样按 train-version 复用。

这项优化不改变 task 筛选、train/test index、LabelEncoder 类别顺序或唯一标签
键顺序。sklearn 固定 C 回归测试要求 hard labels/metrics 完全一致，probability
使用 `atol=1e-8`。

## 10. 外部 manifest 与来源校验

`--manifest-path` 允许训练结果写入新目录，同时复用其他目录中的 manifest。
训练开始前会检查：

- manifest case ID 是否全部存在于当前 hidden index；
- shard、offset、hidden size、layer indices、position names 是否一致；
- hidden-state definition 是否为
  `decoder_block_output_pre_final_norm`；
- 请求的 versions、conditions、layers、positions 是否存在；
- manifest 是否包含 V4；
- source experiment、index、dataset、manifest fingerprints 是否匹配。

旧 manifest 没有 fingerprint 时不会跳过检查，而是执行逐 case
`legacy_exhaustive` validation，并把计算出的 fingerprints 写入 run config。

## 11. 输出保护与 resume

训练默认拒绝在已有运行目录中覆盖 config、split、metrics 或 predictions。

同一配置的中断运行可追加：

```text
--resume
```

语义如下：

- immutable config fingerprint 完全相同才允许继续；
- 已完整结束且正式输出齐全时幂等退出；
- 中断且尚无正式 metrics/predictions 时复用 split，从头安全重试；
- 当前不保存模型级 checkpoint；
- 配置不同、split 不一致或已有不完整正式输出时立即失败；
- 只清理当前 attempt 创建的临时文件，不删除其他旧 `*.tmp`。

旧版运行目录没有新版 config fingerprint，不能直接 `--resume`；应使用新的
`--output-dir`。

## 12. 输出文件与字段

| 文件 | 内容 |
| --- | --- |
| `text_only_labels.jsonl` | Text-only 行为标签及 raw 输出 |
| `image_only_labels.jsonl` | Image-only 行为标签及 raw 输出 |
| `label_failures.jsonl` | 每次标签失败或冲突详情 |
| `probe_manifest.jsonl` | Probe records、eligibility 和 hidden reference |
| `manifest_summary.json` | 数据统计及 provenance fingerprints |
| `split_assignments.json` | 唯一 `item_id -> fold` assignment |
| `run_config.json` | immutable config、依赖版本、状态和性能统计 |
| `layer_probe_metrics.json` | 逐 fold/task/layer/setting/subset 指标 |
| `layer_probe_predictions.jsonl` | 逐 case label、预测和完整类别概率 |
| `probe_summary.json` | 有效 folds 的不加权 mean/std 汇总 |

逐 fold 指标包括：

- accuracy、balanced accuracy、macro F1、probability cross-entropy；
- sample/item count；
- train majority class 及其 test accuracy；
- selected C、C selection/fallback 详情；
- permutation accuracy mean/std；
- invalid 或 empty 状态及原因。

Torch fit diagnostics 包括：

- backend、device、C；
- iterations、closure evaluations、retry count；
- final loss、gradient norm、relative loss change、converged；
- preprocessing、GPU transfer 和 fit 时间。

运行级 performance 包括：

- `feature_loading_seconds`；
- `gpu_transfer_seconds`；
- `fit_seconds`；
- `evaluation_seconds`；
- `total_seconds`；
- `fit_count` 与 `fit_count_by_model_type`；
- `mean_iterations`、`p95_iterations`、`non_converged_count`；
- feature/model cache 计数。

跨 fold 汇总只包含 valid folds，使用不加权 mean/std，标准差为总体标准差
`ddof=0`。

## 13. 测试

```bash
cd /root/autodl-tmp
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest layer_metacognition/probe/tests -q
```

测试覆盖：

- 单层/多层 shard、named positions、FP32 输出、cache 和 integrity error；
- 标签去重/冲突、partial manifest、hard/easy eligibility；
- grouped outer/inner split、跨版本 item 隔离和 unseen test class；
- 指标、概率类别顺序与 unique-key permutation；
- sklearn 模型复用的无损等价性；
- Torch 多分类、Conflict 单-logit、100+100 retry 与 NaN/Inf；
- sklearn/Torch 相同 fixed-C parity；
- provenance、外部 manifest、output protection 和 resume；
- CPU 集成测试及 CUDA 可用时的 GPU smoke。

测试使用临时目录和合成 tensor，不加载真实 Qwen 模型，也不会修改正式实验
结果。
