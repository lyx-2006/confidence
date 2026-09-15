# Qwen3-VL delayed-SA Steering review

本目录把 `dp_SA` 的 Qwen2.5-VL delayed-SA capture/Steering 实验迁移到
`Qwen/Qwen3-VL-8B-Instruct`。旧 `dp_SA` 不做修改；新实验使用 Qwen3 自己生成的
Phase 0 答案、clean soft-SA、construction/test 划分和 Steering 向量。

## 与 Qwen2.5-VL 推理实现的关键差异

| 项目 | Qwen2.5-VL-7B | Qwen3-VL-8B | 对本实验的处理 |
|---|---:|---:|---|
| 文本 decoder | 28 层，hidden 3584 | 36 层，hidden 4096 | Qwen3 重新 capture/构造向量，禁止复用旧 hidden |
| 视觉编码 | ViT depth 32，patch 14 | ViT depth 27，patch 16 | 每条样本都用 Qwen3 processor 重新展开视觉 token |
| 视觉融合 | 输入 embedding 替换 | 输入替换并通过 DeepStack 向早期语言层追加多尺度视觉特征 | capture 从 layer 8 开始，已越过三次早期 DeepStack 注入 |
| RoPE | M-RoPE | Interleaved-MRoPE，原生 256K context | 完全交给 Qwen3 forward 生成 position ids |
| attention | 无 Q/K norm | Q/K RMSNorm | Steering 仍作用于完整 decoder block output |
| decoder 返回 | tuple，首项为 hidden | 直接返回 Tensor | `hooks.py` 显式兼容两种结构 |
| Transformers 类 | `Qwen2_5_VLForConditionalGeneration` | `Qwen3VLForConditionalGeneration` | 从本地 `interface.py` 加载 `Qwen3VLInference` |
| chat template | 无 system 时自动补 `You are a helpful assistant.` | 不自动补 system | 按约定使用 Qwen3 原生模板，不人为补 system |

官方资料：[Qwen3-VL 模型卡](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)、
[Transformers Qwen3-VL 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)。

本仓库实测同一 delayed-SA 样本经两套 processor 后，Qwen3 序列长度为 1470、
Qwen2.5 为 1826；五个语义 token 的相对顺序保持不变，但绝对 index 全部变化。
因此 capture 和 Steering 都会从当前 rendered prompt 动态定位 `LAT`、`PANL`、
`PANL+1`、`CLE`、`SAC`，不会读取历史 token index。

## 目录和产物

```text
qwen3 review/
├── README.md
├── capture/                  # 默认 clean capture 输出
└── Steering/
    ├── capture.py            # Phase 0 + clean hidden capture
    ├── steering.py           # 向量构造 + 参数化干预
    ├── config.py             # 固定实验契约
    ├── contracts.py          # CLI/指纹/hidden key 校验
    ├── hooks.py              # layer 8..35 精确 capture
    ├── runtime.py            # Qwen3 专用动态加载
    └── tests/
```

`capture/hidden/*.npz` 中每个成功 case 固定包含 140 个 float16 数组：五个位置乘
layer 8–35，每个数组 shape 为 `[4096]`。完整 1656-case capture 预计约 1.8 GiB。
大体积产物由本目录自己的 `.gitignore` 排除，不改根目录忽略规则。

## 运行

开始实验前先确认本地模型完整并校验：

```bash
python qwen-3-vl/interface.py verify --model-dir qwen-3-vl/model
```

小规模 capture（至少 20 个 item-disjoint case 才能执行后续 smoke Steering）：

```bash
python "qwen3 review/Steering/capture.py" \
  --max-samples 20 \
  --unique-items \
  --output-root "qwen3 review/capture-smoke"
```

正式 capture：`--max-samples` 按 case 行数计数；例如 500 表示处理 500 条 case，
不是 500 个唯一 item：

```bash
python "qwen3 review/Steering/capture.py" --max-samples 500
# 中断后继续
python "qwen3 review/Steering/capture.py" --max-samples 500 --resume
```

Steering 的三个实验维度均为必填多值参数，并运行笛卡尔积。例如：

```bash
python "qwen3 review/Steering/steering.py" \
  --positions LAT PANL CLE 'PANL+1' SAC \
  --layers 8 12 18 24 30 35 \
  --alphas -10 -2 0 2 10 \
  --output-root "qwen3 review/Steering/output/main"
```

使用小规模 capture 时加 `--smoke` 并显式指定其路径：

```bash
python "qwen3 review/Steering/steering.py" \
  --capture-root "qwen3 review/capture-smoke" \
  --output-root "qwen3 review/Steering/output/smoke" \
  --positions PANL --layers 8 --alphas -2 0 2 --smoke
```

相同 output root 只有在模型、capture 和参数网格指纹完全一致时才能 `--resume`；
更换任一参数请使用新的 output root。

正式样本划分保持两极 construction 各 25 条；由于当前 500 条 capture 的文本侧
item-disjoint 候选只有 31 条，Steering 测试集采用图像侧 50 条、文本侧最多 31 条，
两侧与 construction 均不共享 item。向量为 `mean(high_image) - mean(high_text)`，
并按平均 residual norm 的 3% 缩放。

## 验证

```bash
pytest -q "qwen3 review/Steering/tests"
```

CPU/processor 测试覆盖参数范围、140-key contract、Tensor/tuple hook、3% 方向归一化、
指纹 resume、Qwen3 原生模板和五位置定位。真实 GPU smoke 通过上述两条 smoke 命令执行；
Steering 会强制检查 hook 只注入一次，并在 `alpha=0` 时检查 clean logits/probabilities
最大误差不超过 `1e-6`。
