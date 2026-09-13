# gemma-3-4b-it 本地推理

`google/gemma-3-4b-it` (4.30B, 多模态) 的本地 BF16 推理环境。

```
gemma-3-4b-it/
├── models/          # 模型权重与配置 (8.64 GB, 已通过 SHA256 校验)
├── interface.py     # 推理接口, 强制 BF16
├── download.py      # 下载器 (30 进程分片, 支持断点续传)
├── test_image.png   # 多模态冒烟测试用图 (红圆 / 蓝方 / 绿三角)
└── logs/            # 下载与升级日志 (已 gitignore)
```

## 用法

```bash
cd /root/autodl-tmp/gemma-3-4b-it

# 纯文本
python3 interface.py -p "用一句话解释什么是梯度下降"

# 图文
python3 interface.py -p "图里有哪些几何形状?" -i test_image.png

# 采样解码
python3 interface.py -p "用三个词形容秋天" --temperature 0.7 --top-p 0.9
```

代码里调用:

```python
from interface import Gemma3

m = Gemma3()
m.chat("你好")
m.chat("图里有什么?", image="cat.jpg")
m.multi_turn([("user", "1+1=?"), ("assistant", "2"), ("user", "再乘 3 呢?")])
```

## 环境约束 (重要)

**图片推理需要 `torch>=2.6`。** Gemma3 的图像块双向注意力掩码走 `torch.vmap`,
而 torch 2.5 的 vmap 不支持该掩码函数内部的张量索引, 会直接抛错。

本机已从 2.5.1+cu124 升级到 **2.6.0+cu124** (torchvision 0.21.0+cu124) 以满足该要求。
若其他实验需要回退:

```bash
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124
```

回退后纯文本推理仍然可用, 图片推理会报
`Using or_mask_function or and_mask_function arguments require torch>=2.6`。

模型中 `eos_token_id` 为 `[1, 106]`, 原生 `torch_dtype` 就是 `bfloat16`。
`interface.py` 加载后会把所有浮点权重扫一遍, 出现任何非 BF16 权重即拒绝运行。

## 重新下载

下载源是 ModelScope (`google/gemma-3-4b-it`, 无需授权; HF 上该模型是 GatedRepo)。
已完成的分片在 `models/.chunks/` 留下标记, 重跑会跳过, 中断后直接再执行即可:

```bash
python3 download.py --workers 30 --chunk-mb 32   # 默认值
python3 download.py --verify-only                # 只校验 SHA256
```
