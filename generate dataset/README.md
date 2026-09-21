# Shape–Color 数据生成

## 目录

```text
generate dataset/
├── code/
│   ├── current/          # 当前 image pool、文本校准和测试集划分代码
│   │   └── tests/
│   └── legacy/           # 旧版 text/image 配对生成代码
│       └── tests/
└── datasets/
    ├── current/          # 当前正式数据
    └── legacy/           # 旧数据、探索实验与中间产物
```

当前测试数据是 `datasets/current/conflict_test.json`，图片池是
`datasets/current/interval_pool_full`。测试 JSON 中的 image 路径相对于
`datasets/current`，移动整个 `current` 目录时仍然有效。

正式 Accepted 图片的完整 layout 保存在各颜色/形状 JSON 的 `layout` 字段中；
探索实验的 layout 继续以独立 `layout.json` 或 `.layout.json` 保存。整理过程不删除 layout。

## Qwen3 难度探索自动化

`explore_image_difficulty.py` 用于运行固定的 22 轮难度实验：先测试 1、4、7、10
个图形的无遮挡场景，再对 4、7、10 图形逐级测试 10%–60% 的目标遮挡。每轮固定
3 色 × 3 图形 × 3 个 seed，并测试 blur `0,2,4,8,12,16,24,32`，共 216 张；
主实验合计 4752 张。

```bash
python "generate dataset/code/current/explore_image_difficulty.py" run

# 中断后恢复；配置指纹不一致时会拒绝继续
python "generate dataset/code/current/explore_image_difficulty.py" run --resume
```

默认输出到 `generate dataset/datasets/legacy/experiments/image_difficulty_experiments`。每轮目录包含
`config.json`、`round_NNN.md`、`candidate_results.jsonl`、`summary.json`、
`contact_sheet.png` 以及可复现的 layout、mask、sharp 和 blur 图片。每轮完成后会原子更新
`experiment_summary.csv` 和 `SUMMARY.md`。

阈值只从生成答案和 restricted top-1 同时正确的样本中搜索，并要求四档各至少 20 张、
覆盖至少 6 个 color × shape 组合、median blur 不下降。如果主实验不满足条件，程序会在
最高可用 Entropy 配置附近最多增加 3 轮新 seed。模型连续异常三次时会停止运行，避免把
环境错误登记成图片错误。

### 5 轮极端遮挡测试

`run_extreme_difficulty.py` 独立测试 10–12 个语义物体、60%–80% 目标遮挡、2–4 个
联合遮挡物，并为每个场景叠加 blur `0,2,4,8,12,16,24,32`。每张图片运行三次
Qwen3 image-only；单次必须同时通过生成答案和 restricted top-1，至少两次正确才通过。

```bash
python "generate dataset/code/current/run_extreme_difficulty.py" run

# 中断恢复
python "generate dataset/code/current/run_extreme_difficulty.py" run --resume
```

默认输出目录为 `generate dataset/datasets/legacy/experiments/image_difficulty_extreme`。三次完整测量均写入
`candidate_results.jsonl`，最终联合统计位于 `SUMMARY.md` 和 `aggregate_summary.json`。

## Qwen3 image pool（独立入口）

`generate_image_pool.py` 生成不带文本先验、不区分 consistent/conflict 的纯图片池。
场景复杂度、遮挡和 Gaussian blur 只用于产生候选，最终难度由 Qwen3 在固定
image-only prompt 下的 12 色 normalized entropy 分档。正式图片必须同时满足生成答案
和 restricted top-1 均等于真实颜色。

先运行固定的 324 张 pilot，再分析并人工确认建议阈值：

```bash
python "generate dataset/code/current/generate_image_pool.py" pilot
python "generate dataset/code/current/generate_image_pool.py" analyze
```

阈值确认后进行正式生成或导入旧图片；三个阈值均为 0–1 normalized entropy：

```bash
python "generate dataset/code/current/generate_image_pool.py" build \
  --thresholds 0.20,0.40,0.60 \
  --quota-per-level 10

python "generate dataset/code/current/generate_image_pool.py" import-legacy \
  --thresholds 0.20,0.40,0.60 \
  --quota-per-level 10 \
  --resume
```

示例阈值仅说明 CLI 格式，不能在 pilot 分析和人工 contact sheet 检查前直接作为正式标准。
默认模型目录为 `qwen-3-vl/model`，pilot 和正式池分别写入
`generate dataset/datasets/legacy/experiments/image_pool_pilot` 与
`generate dataset/datasets/current/image_pool`。
恢复已有运行必须使用 `--resume`，且模型、prompt、profile、blur、阈值、seed 和配额配置
必须与首次运行一致。

每个颜色一个目录，每个 shape 的 JSON 顶层为数组，图片统一命名为
`shape_color_六位编号.png`。`candidate_results.jsonl` 是完整、可恢复的候选测量账本，
`rejected.jsonl` 保存精简拒收原因；同一个 `base_scene_id` 的 blur 变体在后续数据划分时
必须放在同一个 split。

### 12×17 Entropy 区间池完整运行

`run_interval_pool_full.py` 覆盖全部 12 色 × 17 图形，共 204 个 `color × shape`
组合。默认同时激活 12 个组合，使用 24 个 CPU 生成进程和最多 48 个待处理任务；
任务按 construction→pair 交错提交，使第一波 worker 覆盖 12 个不同组合。Qwen3
仍只加载一个实例并逐张测量，生成进程会在 GPU 推理期间继续准备后续图片。

先查看计划或只读状态，不会创建正式输出：

```bash
python "generate dataset/code/current/run_interval_pool_full.py" plan
python "generate dataset/code/current/run_interval_pool_full.py" status
```

正式运行与恢复命令：

```bash
python "generate dataset/code/current/run_interval_pool_full.py" run
python "generate dataset/code/current/run_interval_pool_full.py" run --resume

# 两个独立 Qwen worker，各自固定到一张 GPU
python "generate dataset/code/current/run_interval_pool_full.py" run --resume --gpu-devices 0,1
```

正式运行默认先从 `datasets/interval_pool_pilot` 导入已完成的候选账本和 accepted PNG。
导入前会核对源配置指纹、模型、prompt、construction、图片 SHA256 和 attempt 上限；
导入是幂等的，不调用 Qwen。已有三个组合会继承全部 2377 次测量及各区间状态，
已经达到配额或 200 次预算的区间不会重复运行。使用 `--no-reuse` 可显式关闭复用。
正式输出目录为 `datasets/interval_pool_full`。

正式运行把 `_staging` 作为临时目录。Accepted 永久保留最终 PNG、shape JSON、完整
layout、layout SHA256 和 12 色 logits/probabilities。当前正式池完成整理后已删除大型候选账本
和失败索引，并把残留 staging 移入 `datasets/legacy/intermediate`；它应作为只读数据使用，
不能再依赖原目录直接 `--resume`。如需继续生成，请指定新的 output root。

旧 Accepted 若缺少 layout，恢复启动时会根据 seed、shape、color、物体数和遮挡配置
确定性重建并原子回填。回填结果写入 `layout_sha256`；已有 pilot 样本已通过与原始
`scene/layout.json` 的逐一哈希一致性测试。

`--gpu-devices 0,1` 使用一个主调度器和两个独立的 Qwen 模型进程，推理结果并行计算，
但仍由主进程按稳定任务顺序写入唯一账本。不要同时启动两份完整运行命令指向同一输出目录。
GPU worker 数属于运行时吞吐配置，不改变图片、Entropy 或配额配置，因此单卡任务可在候选终态
边界停止后，用相同输出目录和 `--resume --gpu-devices 0,1` 安全迁移为双卡。

## 概述

生成**可恢复的、纯图像的形状×颜色视觉推理数据集**。

输入数据集中已有的 (shape, color) 组合会被跳过，仅生成缺失的组合。每个组合生成 4 张图像（easy/hard × consistent/conflict）。DeepSeek 只返回受限的视觉风格 JSON；仓库内可信 renderer 解析本地 RNG layout 并直接生成 17 种图形、图片和 masks，不执行任何模型生成代码。最后由本地 Qwen2.5-VL 验证图像是否可被正确回答。

输出数据集**故意不包含 `selected_text_priors`**，以避免文本先验泄漏。

---

## 参数表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input-dataset` | `str` | `datasets/dataset_test.json` | 现有输入数据集路径，用于发现已有的 shape-color 组合并复用其 `irr`/`null` 图片 |
| `--prior-pool` | `str` | `datasets/color_prior_pool.json` | 颜色先验池 JSON 文件，必须包含全部 12 种颜色且每种至少有一个 accepted 的非空 `text_clue` |
| `--output-dataset` | `str` | `generate dataset/datasets/legacy/original/generated_shape_color_dataset.json` | 输出数据集 JSON 路径，每个完成的组合会原子写入 |
| `--image-dir` | `str` | `generate dataset/datasets/legacy/original/generated_shape_color_images` | 渲染图片输出目录，结构为 `{id}_{branch}_{difficulty}.png` 等 |
| `--model-path` | `str` | `qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct` | Qwen2.5-VL 模型权重目录路径，用于 `ExtendedQwenVLInference` |
| `--seed` | `int` | 随机生成 (64-bit) | 随机种子，决定形状顺序、冲突颜色映射、布局生成。同一 seed 保证完全可复现。必须 ≥ 0 |
| `--workers` | `int` | `16` | consistent/conflict branch/DeepSeek 并发进程数，范围 1–64；每个进程内部严格先 easy、后 hard |
| `--gpu-queue` | `str` | 输出文件同名 `.gpu_queue.json` | 持久化 FIFO GPU 等待队列；所有 worker 只入队，不加载模型 |
| `--gpu-wait-timeout` | `float` | `86400` | worker 等待本地模型测试结果的最长秒数 |
| `--resume` | `flag` | `False` | 从中断处恢复运行。需要 `.state.json` 文件存在且配置匹配 |
| `--recreate` | `flag` | `False` | 仅重建 `invalid_datasets` 中的 conflict-hard；每张图一个 worker，最多 20 次，成功后追加至 `valid_datasets` |
| `--dry-run` | `flag` | `False` | 试运行模式：执行一次完整的 easy consistent 流水线（真实 API 调用 + 模型推理），打印结果但不持久化任何文件 |

### 互斥约束

- `--resume` 和 `--dry-run` **不能同时使用**
- `--recreate` 和 `--dry-run` **不能同时使用**
- `--resume` 时会忽略 `--seed`（自动使用 state 中保存的种子），若显式传入且不匹配会报错
- 首次运行时若 `--output-dataset` 或 `.state.json` 已存在会拒绝覆盖，需选择新路径或使用 `--resume`

---

## 运行方式

### 1. 环境要求

```bash
pip install openai Pillow
```

**依赖文件**（脚本同目录或项目根目录下）：

| 文件 | 用途 |
|------|------|
| `api_config.json` | DeepSeek API 配置：`{"api_key": "...", "base_url": "..."}` |
| `datasets/dataset_test.json` | 输入数据集 |
| `datasets/color_prior_pool.json` | 12 种颜色的先验池 |
| `confidence_test/inference_extension.py` | VLM 推理扩展模块 |
| `--model-path` 目录 | Qwen2.5-VL 模型权重 |

### 2. 运行测试

```bash
python -m pytest "generate dataset/code/legacy/tests/test_generate_shape_color_dataset.py" -v
```

### 3. 试运行

```bash
python "generate dataset/code/legacy/generate_shape_color_dataset.py" --dry-run
```

### 4. 正式运行

```bash
python "generate dataset/code/legacy/generate_shape_color_dataset.py" \
  --workers 16 \
  --seed 42 \
  --input-dataset datasets/dataset_test.json \
  --prior-pool datasets/color_prior_pool.json \
  --output-dataset datasets/generated_shape_color_dataset.json \
  --image-dir datasets/generated_shape_color_images \
  --model-path qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct
```

### 5. 中断恢复

```bash
# Ctrl+C 中断后，直接使用 --resume 继续
python "generate dataset/code/legacy/generate_shape_color_dataset.py" --resume
```

收到 `Ctrl+C` 时，父进程会取消等待任务、终止并回收所有 branch worker；被中断的 GPU
队列任务会在下次 `--resume` 时重新入队。

### 6. 重建失败的 conflict-hard

```bash
python "generate dataset/code/legacy/generate_shape_color_dataset.py" --recreate

# Ctrl+C 后恢复
python "generate dataset/code/legacy/generate_shape_color_dataset.py" --recreate --resume
```

重建模式固定读取 `datasets/invalid_datasets/generated_shape_color_dataset.json`，仅复用
conflict-easy 的 layout 和 normalized entropy。每轮 hard 都重新排布 distractor 和遮挡方向，Qwen
仍串行测试 3 次；只有三次答案都等于 `conflict_answer` 且
`hard_normalized_entropy - easy_normalized_entropy > 0.25` 才发布。失败原因和完整 layout JSON 会先交给
DeepSeek Failure Analyst，建议再传给下一轮 Generation Agent。每张待重建图片各占一个 worker。

成功样本会把四组图片、layout 和 masks 统一重编号并复制到 `valid_datasets/images`，随后原子追加到
`valid_datasets/generated_shape_color_dataset.json`。状态、worker 检查点和 GPU 队列分别使用
`.recreate.state.json`、`.recreate.branches/` 和 `.recreate.gpu_queue.json`。

valid 发布及 16 个 artifact 完整性复核通过后，对应源 item 和 artifact 会立即从 invalid 中删除。
当本轮所有 worker（包括失败项）都结束后，剩余 invalid 会按当前顺序压缩编号为 `001..N`，同步更新
四组 image 路径，并进入新的 recreate cycle；下一轮使用 `--recreate --resume` 即可继续处理重编号后的样本。

当前 invalid 目录有 17 条整理时发生的 asset/标签错配；重建预检以
`question + answer + conflict_answer` 为标签基准，从原始汇总唯一恢复正确 artifact，且不会改写 invalid 文件。

尺寸或 renderer 配置更新时，已有 attempt、已锁定 easy 和已生成图片保持不变；新尺寸范围只应用于后续新候选。

---

## 主要硬编码常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `CANVAS_SIZE` | 1024 | 画布尺寸 (px) |
| `EASY_MAX_ATTEMPTS` | 5 | easy 布局最大尝试次数 |
| `HARD_MAX_ATTEMPTS` | 10 | hard 布局最大尝试次数 |
| `RECREATE_HARD_MAX_ATTEMPTS` | 20 | recreate conflict-hard 最大尝试次数 |
| `ENTROPY_GAP_THRESHOLD` | 0.25 | hard 候选的 entropy gap 严格下限（不含等于） |
| `OCCLUSION_RANGE` | (0.70, 0.80) | hard 场景目标遮挡比例范围 |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash-aistar` | DeepSeek API 模型名 |
| DeepSeek Planner temperature | `0.2` | 只生成受限 `render_style` JSON，不生成或校验 Python |
| 最大图形 bbox | `250×250` | 本地硬校验，任一图形宽或高超过 250 px 即拒绝 |
| easy 图形尺寸 | target/distractor `180–250` | `size` 表示 bbox 最大边 |
| hard 新增图形尺寸 | `180–250` | occluder 从 target mask 求解，最大边同样不得超过 250 px |
| `COLORS` | 12 种颜色 | red, orange, yellow, green, blue, cyan, purple, pink, brown, white, black, gray |
| `SHAPES` | 17 种形状 | rectangle, square, parallelogram, trapezoid, diamond, circle, oval, semicircle, crescent, triangle, pentagon, hexagon, octagon, star, heart, arrow, cross |

---

## 输出文件结构

```
generate dataset/datasets/legacy/original/
├── generated_shape_color_dataset.json   # 主输出
├── generated_shape_color_dataset.state.json  # 恢复状态
├── generated_shape_color_dataset.gpu_queue.json  # FIFO GPU 测试队列
├── generated_shape_color_dataset.branches/  # branch 独立检查点
└── generated_shape_color_images/
    ├── 121_consist_easy.png
    ├── 121_consist_easy.layout.json
    ├── 121_consist_easy.target_mask.png
    ├── 121_consist_easy.occluder_mask.png
    ├── 121_consist_hard.png
    ├── 121_consist_hard.layout.json
    ├── ...
    ├── 121_conflict_easy.png
    ├── ...
    └── 122_consist_easy.png ...
```

---

## 架构流程图

```
build_manifest()
  ├── 扫描输入数据集 → 去重得到已有组合
  ├── 随机排列形状顺序
  ├── 构建冲突颜色错排 (cyclic derangements)
  └── 根据当前 SHAPES × COLORS 生成缺失组合列表

父进程:
  ├── 只加载一次 Qwen
  ├── 启动最多 64 个 branch worker process
  └── 按 FIFO 顺序消费 .gpu_queue.json，每张图串行测试 3 次

每个 branch worker（consistent 或 conflict）:
  ├── easy: DeepSeek render_style JSON → JSON schema 收窄 → trusted local renderer
  ├── 本地 renderer 直接生成 RGB 图片、target mask、occluder mask 和 layout
  ├── hard 遮挡位置由本地 target mask 求解，实测比例必须为 70%–80%
  ├── 写入 GPU queue 并等待测试结果
  ├── 失败结果反馈 Planner，仅调整下一候选的受限视觉风格
  └── easy 锁定后才开始 hard；hard 使用相同队列和反馈流程

父进程收到同一 item 的两个 branch 结果后:
  └── 按 manifest 顺序原子写入 output dataset
```
