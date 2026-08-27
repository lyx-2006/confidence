# generate_shape_color_dataset.py

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
| `--output-dataset` | `str` | `generate dataset/datasets/generated_shape_color_dataset.json` | 输出数据集 JSON 路径，每个完成的组合会原子写入 |
| `--image-dir` | `str` | `generate dataset/datasets/generated_shape_color_images` | 渲染图片输出目录，结构为 `{id}_{branch}_{difficulty}.png` 等 |
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
python -m pytest "generate dataset/test_generate_shape_color_dataset.py" -v
```

### 3. 试运行

```bash
python "generate dataset/generate_shape_color_dataset.py" --dry-run
```

### 4. 正式运行

```bash
python "generate dataset/generate_shape_color_dataset.py" \
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
python "generate dataset/generate_shape_color_dataset.py" --resume
```

收到 `Ctrl+C` 时，父进程会取消等待任务、终止并回收所有 branch worker；被中断的 GPU
队列任务会在下次 `--resume` 时重新入队。

### 6. 重建失败的 conflict-hard

```bash
python "generate dataset/generate_shape_color_dataset.py" --recreate

# Ctrl+C 后恢复
python "generate dataset/generate_shape_color_dataset.py" --recreate --resume
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
generate dataset/datasets/
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

## V2 入口与输出

本脚本不传 `--recreate`/`--legacy`/`--dry-run` 时即走 V2 图像 producer
（`_run_v2_cli`），默认输出到 `generation_v2_outputs/formal/image/`；旧目录
只读且不会被覆盖。V2 默认 `--branches conflict`，可显式启用
`conflict,consistent`；每个 difficulty 由 `--images-per-difficulty`（1–32）
及 conflict easy/hard 覆盖控制，hard 数量不能大于 easy。每个分支的 easy/hard
均为 variant 数组，artifact 文件名为 `{id}_{branch}_{difficulty}({variant_index}).*`。

图像 producer 默认保持 `--workers`（1–64，默认 16）个分支 worker 进程并行，
每个进程负责一个 manifest item 的 easy→hard；与文本颜色池入口共用
`generation_runtime.py` 的 Qwen batch 运行时——所有生成进程通过持久化队列
共享父进程中唯一的 Qwen 实例，scheduler 任意时刻只执行一个 batch。
DINOv2 默认要求本地 `--similarity-model-path`，只有显式
`--download-similarity-model` 才下载到隔离目录。`--resume` 会校验完整配置；
旧 split/refine/recreate 工具遇到 `shape_color_dataset.v2` 会直接报 schema guard
错误。
