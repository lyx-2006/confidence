# V3/V4 Joint Hidden State Linear Probe

本目录用于在已有 V3/V4 joint 实验的逐层 hidden state 上训练线性 Probe，研究模型内部是否存在可线性解码的单模态答案信息。

Probe 只读取现有实验结果，不会重新运行 joint inference，也不会修改：

- `layer_metacognition/output/v3_v4/results.jsonl`
- `layer_metacognition/output/v3_v4/hidden_states/index.json`
- 已有 hidden-state shard

所有 Probe 代码位于 `layer_metacognition/probe/`，生成结果统一保存到：

```text
layer_metacognition/output/v3_v4/probe/
```

## 1. 实验目标

本实验使用三类模型行为答案：

- \(A_T\)：模型只看到 Question + Text clue 时生成的答案。
- \(A_I\)：模型只看到 Question + Image 时生成的答案。
- \(A_J\)：现有 V3/V4 joint run 中的 `generated.current_answer`。

所有标签均来自模型自身输出，并使用仓库现有的答案规范化逻辑。数据集 ground truth、`answer`、`text_ans`、`conflict_ans` 以及根据 condition 人工推断的答案都不会用作 Probe 标签。

实验读取两个 token 位置：

- **AC**：当前答案输出前，answer marker 对应的 cognition token。
- **PANL**：当前答案已经输出后，紧接答案字段的换行 token。

四个 Probe 任务为：

| 任务 | Hidden 位置 | 预测目标 |
| --- | --- | --- |
| `ac_text_answer` | AC | \(A_T\) |
| `ac_image_answer` | AC | \(A_I\) |
| `panl_text_answer` | PANL | \(A_T\) |
| `panl_image_answer` | PANL | \(A_I\) |

每个 Probe 只接收一个 layer、一个位置的 hidden vector。不同层不会拼接，layer index 也不会作为特征。

Linear Probe measures linearly decodable latent answer information. It does not by itself demonstrate causal use of that information.

## 2. 数据范围

Image-only 标签和 Image Probe 严格只使用：

```text
consistent_easy
conflict_easy
```

`consistent_hard`、`conflict_hard`、`null`、`irr` 不会进入 Image Probe 的训练、验证、测试、permuted-label baseline 或汇总指标。

Text Probe 默认同样使用 matched-easy 数据，使 Text 和 Image Probe 的 case 范围一致。辅助分析可以通过：

```text
--text-scope all
```

使用所有具备有效 \(A_T\) 的 case。

## 3. 模型与数据划分

Hidden-state Probe 使用：

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        penalty="l2",
        class_weight="balanced",
        solver="lbfgs",
        max_iter=5000,
    )),
])
```

正则化参数从以下集合选择：

```python
[0.01, 0.1, 1.0, 10.0]
```

数据按 `item_id` 进行 grouped 5-fold cross-validation。同一道题的不同 prior、condition、V3/V4、AC/PANL 和 layer 始终属于同一 fold。

所有任务共享同一份 `split_assignments.json`，并运行四种版本设置：

- `v3_to_v3`：V3 train items → V3 test items
- `v4_to_v4`：V4 train items → V4 test items
- `v3_to_v4`：V3 train items → V4 test items
- `v4_to_v3`：V4 train items → V3 test items

跨版本测试仍然使用不同 item，不会在某个 item 的 V3 hidden state 上训练后直接测试同一 item 的 V4 hidden state。

## 4. 对照实验

### PANL current-answer baseline

PANL 位于当前答案输出之后，因此可能直接包含 \(A_J\) 的答案字符串信息。两个 PANL 任务会额外训练：

```text
current_answer → text_only_answer
current_answer → image_only_answer
```

该 baseline 使用相同的 outer fold、版本设置、训练/测试 item 和分析子集，用于衡量答案泄漏。

### Permuted-label baseline

默认对训练标签运行 20 次固定种子的 permutation。测试标签保持不变，且 permutation 在唯一行为标签键上进行：

- Text：`(item_id, prior_index)`
- Image：`(item_id, condition)`

这样可以避免同一个行为标签的重复 case 被随机赋予互相冲突的标签。

## 5. 安装

在仓库根目录运行：

```bash
python -m pip install -r layer_metacognition/probe/requirements.txt
```

该文件只增加 Probe 训练和测试所需的 `scikit-learn`、`pytest` 等依赖。Qwen 模型运行依赖仍由现有 `qwen-2.5-vl` 环境提供。

## 6. 完整运行流程

### 第一步：生成或恢复单模态标签

```bash
python -m layer_metacognition.probe.generate_unimodal_labels \
  --experiment-dir layer_metacognition/output/v3_v4 \
  --dataset datasets/dataset_with_images.json \
  --model-path qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct \
  --resume
```

脚本优先从现有 V3 `generated.initial_answer` 提取 \(A_T\)。只有缺失 text 标签或尚未完成 easy image 标签时才加载模型，且模型和 processor 在整个进程中只加载一次。

### 第二步：构建 Probe manifest

```bash
python -m layer_metacognition.probe.build_probe_manifest \
  --experiment-dir layer_metacognition/output/v3_v4
```

该步骤合并 \(A_T\)、\(A_I\)、\(A_J\) 与 hidden-state reference，并对照 index 检查 case ID、shard、offset、layers、position names、hidden size 和 hidden-state 定义。

### 第三步：训练逐层 Probe

```bash
python -m layer_metacognition.probe.train_layer_probes \
  --experiment-dir layer_metacognition/output/v3_v4 \
  --layers 19 20 21 22 23 24 25 26 27 \
  --n-splits 5 \
  --seed 42 \
  --text-scope matched_easy
```

### 第四步：汇总结果

```bash
python -m layer_metacognition.probe.analyze_probe_results \
  --experiment-dir layer_metacognition/output/v3_v4
```

所有 CLI 都支持相对路径、绝对路径和 `--help`。

## 7. 输出文件

默认输出目录包含：

| 文件 | 内容 |
| --- | --- |
| `text_only_labels.jsonl` | Text-only 模型行为标签 |
| `image_only_labels.jsonl` | Easy Image-only 模型行为标签 |
| `label_failures.jsonl` | 标签冲突、解析失败和生成失败 |
| `probe_manifest.jsonl` | Probe case、标签和 hidden reference |
| `manifest_summary.json` | Manifest 样本与排除统计 |
| `split_assignments.json` | 唯一的 `item_id → fold` 映射 |
| `layer_probe_metrics.json` | 逐 task/layer/fold/version/subset 指标 |
| `layer_probe_predictions.jsonl` | 逐样本预测与完整类别概率 |
| `probe_summary.json` | 跨有效 fold 的 mean/std 汇总 |
| `run_config.json` | layers、seed、C grid、scope 和依赖版本 |

主要指标包括：

- accuracy
- balanced accuracy
- macro F1
- cross-entropy
- majority-class baseline
- permuted-label accuracy mean/std
- sample count 和 item count

测试标签中若出现训练阶段未见过的类别，整个 fold 会标记为 invalid，不会使用测试标签扩展 LabelEncoder，也不会静默删除测试样本。

## 8. 测试

```bash
python -m pytest layer_metacognition/probe/tests -q
```

测试使用临时目录和合成 tensor，不会加载真实大模型，也不会修改正式实验结果。
