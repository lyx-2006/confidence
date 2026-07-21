# Qwen2.5-VL-7B-Instruct — Native Transformers Inference

基于 Hugging Face Transformers 原生推理模式部署 Qwen2.5-VL-7B-Instruct。

## 环境

| 组件 | 版本 |
|------|------|
| OS | Ubuntu 22.04 |
| Python | 3.12.3 |
| PyTorch | 2.5.1+cu124 |
| CUDA | 12.4 |
| GPU | NVIDIA GeForce RTX 4090 24GB |
| 模型 | Qwen/Qwen2.5-VL-7B-Instruct |

## 项目结构

```
/root/autodl-tmp/qwen-2.5-vl/
├── inference.py              # 推理主程序
├── test_inference.py         # 冒烟测试
├── check_environment.py      # 环境检查
├── requirements.txt          # Python 依赖 (不含 torch/torchvision/flash-attn)
├── README.md                 # 本文件
├── models/
│   └── Qwen2.5-VL-7B-Instruct/   # 模型文件
├── huggingface_cache/        # HF 缓存目录
├── logs/                     # 日志
└── outputs/                  # 输出结果
```

## 安装

```bash
# 设置环境变量
export HF_HOME=/root/autodl-tmp/qwen-2.5-vl/huggingface_cache

# 安装依赖 (不重装 torch/CUDA)
pip install -r requirements.txt
```

## 下载模型

```bash
export HF_HOME=/root/autodl-tmp/qwen-2.5-vl/huggingface_cache

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen2.5-VL-7B-Instruct',
    local_dir='/root/autodl-tmp/qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct',
    local_dir_use_symlinks=False,
    resume_download=True,
)
"
```

## 环境检查

```bash
cd /root/autodl-tmp/qwen-2.5-vl
python check_environment.py
```

## 推理命令

### 纯文本推理

```bash
python inference.py \
  --prompt "请简单介绍一下你自己。" \
  --max-new-tokens 128
```

### 图像推理

```bash
python inference.py \
  --image /root/autodl-tmp/test.jpg \
  --prompt "请描述这张图片。" \
  --max-new-tokens 128
```

### 指定模型路径

```bash
python inference.py \
  --model-path /root/autodl-tmp/qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct \
  --image /root/autodl-tmp/test.jpg \
  --prompt "图中主要有什么？"
```

### 保存结果

```bash
python inference.py \
  --image test.jpg \
  --prompt "图中有什么？" \
  --output outputs/result.json
```

## 输出格式

```json
{
  "model_path": "/root/autodl-tmp/qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct",
  "image_path": "/root/autodl-tmp/test.jpg",
  "prompt": "图中有什么？",
  "response": "图中有一张...",
  "generation_config": {
    "do_sample": false,
    "max_new_tokens": 128,
    "use_cache": true
  },
  "runtime": {
    "device": "cuda:0",
    "dtype": "bfloat16",
    "elapsed_seconds": 3.456,
    "peak_gpu_memory_gb": 18.234
  }
}
```

## 测试

```bash
# 自动生成测试图片并运行测试
python test_inference.py

# 使用自定义测试图片
python test_inference.py --image /root/autodl-tmp/test.jpg

# 指定模型路径
python test_inference.py --model-path /path/to/model
```

测试结果保存到 `outputs/smoke_test.json`。

## 关键配置

- **`do_sample=False`** — 确定性贪心解码，不使用 temperature/top_p/top_k
- **`device_map="auto"`** — 自动设备分配
- **`attn_implementation="eager"`** — 标准注意力，支持 hidden states/attention 提取
- **`torch.bfloat16`** — BF16 精度
- **不使用** flash-attn, vLLM, Ollama, pipeline(), bitsandbytes, GPTQ, AWQ

## 常见错误排查

### 1. `KeyError: 'qwen2_5_vl'`

Transformers 版本过旧，需要 >= 4.46.0：

```bash
pip install -U "transformers>=4.46.0"
```

### 2. 找不到 `Qwen2_5_VLForConditionalGeneration`

检查 transformers 版本：

```bash
pip show transformers
python -c "from transformers import Qwen2_5_VLForConditionalGeneration; print('OK')"
```

### 3. `qwen_vl_utils` 未安装

```bash
pip install -U qwen-vl-utils
```

### 4. CUDA 不可用

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

确保 PyTorch 版本带 CUDA：`pip show torch` 应显示 `+cu124`。

### 5. CUDA OOM

RTX 4090 24GB 下 BF16 模型约 14GB，加上 KV cache 和视觉编码器，如遇 OOM：

- 降低 `--max-pixels`（默认 `1280*28*28`）
- 降低 `--max-new-tokens`
- 确认没有其他进程占用 GPU 显存

### 6. 图片读取失败

确认图片路径正确且格式受支持（JPEG/PNG/WebP）：

```bash
python -c "from PIL import Image; img=Image.open('test.jpg'); img.verify(); print('OK')"
```

### 7. 模型下载不完整

检查必要文件：

```bash
ls /root/autodl-tmp/qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct/
```

至少应有：`config.json`, tokenizer 配置, `.safetensors` 权重文件。

### 8. Python 3.12 下 decord 安装失败

`qwen-vl-utils[decord]` 在 Python 3.12 上可能失败，改为安装：

```bash
pip install -U qwen-vl-utils
```

图片推理不需要 decord。

### 9. `output_attentions=True` 与非 eager attention 不兼容

确保使用 `attn_implementation="eager"`。flash_attention_2 和 sdpa 不支持输出 attention 权重。

### 10. `do_sample=False` 时 temperature 参数警告

当 `do_sample=False` 时不要传入 `temperature`、`top_p` 和 `top_k`。贪心解码不使用这些参数，传入会触发 UserWarning（本项目的 `inference.py` 已过滤此警告）。

## 研究接口

`QwenVLInference.forward_analysis()` 方法支持提取：

- `logits`
- `hidden_states`（默认开启）
- `attentions`（需显式开启 `output_attentions=True`）

用于 PANL、confidence probe 等研究实验。
