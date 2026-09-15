# Qwen3-VL-8B-Instruct — 本地部署（Transformers 原生推理）

把 `Qwen/Qwen3-VL-8B-Instruct` 完整拉到本地，用 `transformers` 原生方式跑推理。
下载部分是一个 **40 进程分片下载器**，不走 `snapshot_download`
（那是一个进程一条流，4 个大文件最多 4 并发，且无法对单文件分片）。

## 环境

本机实测环境：

| 项 | 版本 |
|---|---|
| GPU | RTX 4090 24GB |
| Python | 3.12.3 |
| PyTorch | 2.5.1+cu124 |
| transformers | 4.57.6（≥4.57 才有 `Qwen3VLForConditionalGeneration`）|
| CPU / 内存 | 16 核 / 1008 GB |

模型 bf16 权重约 16.4 GiB，24GB 显存放得下，还能留出 KV cache 的余量。

### 实测结果

本机跑通后的实测数据（供参考）：

| 项 | 实测 |
|---|---|
| 下载 16.34 GiB / 535 分片 | **26.5 分钟**（聚合 10～38 MB/s，波动大）|
| 模型加载（4 个分片） | ~4 秒 |
| 显存峰值 | 16.42～16.50 GB |
| 纯文本生成 | ~24 tok/s |
| 图片理解（单图） | ~31 tok/s |
| 多图对比（2 图） | ~26 tok/s |

尾巴上有个坑：最后 2 个分片落到了 hf-mirror 的慢节点上，掉到 ~57 / ~81 KB/s，
多花了约 8 分钟。分片在内存里攒够才落盘，所以这段时间进度条完全不动 ——
不是卡死，是看不见。

## 项目结构

```
qwen-3-vl/
├── interface.py         # 下载器 + 推理接口（唯一入口）
├── test_interface.py    # 下载器测试（含逐字节比对）
├── pytest.ini
├── requirements.txt
├── README.md
└── model/               # 模型参数（下载产物）
    ├── config.json
    ├── model-0000{1..4}-of-00004.safetensors
    ├── tokenizer.json / vocab.json / merges.txt
    └── .download_state/ # 断点续传记录，下完自动只剩 .complete 标记
```

## 安装

```bash
# 依赖（不要重装 torch/CUDA）
pip install -r requirements.txt
```

## 下载模型

```bash
cd qwen-3-vl
python interface.py download                    # 40 进程，32MB 分片
```

共 16 个文件 / 16.34 GiB，切成 **537 个分片**，由 40 个进程并发拉取。

常用参数：

```bash
python interface.py download --workers 40        # 进程数（默认 40）
python interface.py download --chunk-mb 16       # 分片大小（默认 32）
python interface.py download --endpoint mirror   # 强制走 hf-mirror.com（直连）
python interface.py download --endpoint hf       # 强制走 huggingface.co（走代理）
python interface.py download --status            # 只看进度，不写任何数据
python interface.py verify                       # 校验已下载文件
```

**端点选择**：默认 `--endpoint auto`，会并发探测两个端点各 4MB，取快的那个
（并发探测，慢的那个不会拖长启动等待）。本机实测单流速度
hf-mirror.com 约 1.5～2.5 MB/s，走本地代理的官方站约 0.3～0.75 MB/s，
相差 3～5 倍且随网络波动，所以 auto 基本都会选 hf-mirror。
走镜像时脚本会**主动绕过** `http_proxy`（国内代理绕镜像反而更慢）；
走官方站时则使用环境里的 `http_proxy`。

**断点续传**：任意时刻 Ctrl-C 或断网，重新执行同一条命令即可。
每个文件在 `.download_state/` 下有一份 `.done` 记录，记着已完成的分片偏移。

**校验**：全部下完后自动按 HF 的 LFS 元数据逐文件校验 sha256
（小文件没有 sha256，退化为校验大小），通过后写 `.complete` 标记；
下次运行看到 `.complete` 就整文件跳过，不重复校验。
校验不通过的文件不会被标记，重跑即可只补该文件。

**磁盘占用**：文件按最终大小预分配（稀疏），各进程用 `os.pwrite`
直接写自己的偏移，**不需要「先存分片再合并」**，
所以峰值占用就是模型体积本身（约 16.4 GiB），不会翻倍。
注意本机 `/root/autodl-tmp` 可用空间较紧，下载前先确认 `df -h` 有 ~18GB 余量。

## 推理命令

```bash
# 纯文本
python interface.py infer --prompt "介绍一下你自己"

# 单图
python interface.py infer --image img.jpg --prompt "描述这张图片"

# 多图（--image 可重复传）
python interface.py infer --image a.jpg --image b.jpg --prompt "这两张图有什么不同"

# 控制生成长度与视觉分辨率
python interface.py infer --image img.jpg --prompt "图里有什么？" --max-new-tokens 512
python interface.py infer --image img.jpg --prompt "图里有什么？" --max-pixels 1048576

# 保存结果 JSON
python interface.py infer --image img.jpg --prompt "描述这张图片" --output outputs/result.json
```

`min_pixels` / `max_pixels` 默认 **不传**，沿用模型自带
`preprocessor_config.json` 的官方默认值；只有显式传参才会覆盖。

## 研究用途：导出 hidden_states

```bash
python interface.py forward --image img.jpg --prompt "描述这张图片" --output outputs/hidden.pt
python interface.py forward --prompt "1+1=?" --output outputs/hidden.pt --attentions
```

`forward` 默认用 `eager` attention（导出 attention 需要），
返回 `logits` / `hidden_states` / `attentions`，存成 `.pt`。
`--attentions` 显存开销大，默认关闭。

## 输出格式

`infer` 的 JSON 输出：

```json
{
  "model_path": ".../qwen-3-vl/model",
  "image_paths": ["img.jpg"],
  "prompt": "描述这张图片",
  "response": "...",
  "generation_config": {"max_new_tokens": 256, "do_sample": false, "use_cache": true},
  "runtime": {
    "device": "cuda:0",
    "dtype": "bfloat16",
    "attn_implementation": "sdpa",
    "elapsed_seconds": 3.21,
    "new_tokens": 128,
    "tokens_per_second": 39.9,
    "peak_gpu_memory_gb": 17.2,
    "num_images": 1
  }
}
```

## 测试

```bash
pytest test_interface.py -v                  # 全部（需联网）
pytest test_interface.py -v -m "not network" # 只跑离线用例
```

测试重点验证**分片写偏移是否正确** —— 写错偏移在 sha256 校验前不会暴露，
所以用例里用 `curl` 作为独立参照实现，把分片乱序写入的结果与整段下载逐字节比对。

## 下载器工作原理

1. 从 HF API 取文件清单（含每个 LFS 文件的 sha256 和大小）。
2. 按大小升序排列，把每个文件切成 `--chunk-mb` 大小的分片；
   任务列表**跨文件轮转排列**（file1[0], file2[0], …, file1[1], …），
   让 4 个大文件同时推进，避免长尾。
3. 父进程按最终大小预分配所有文件（`ftruncate`，稀疏文件）。
4. 40 个 worker 进程从 `imap_unordered` 取任务：
   发 `Range` 请求 → `os.pwrite` 写到自己的偏移 → 在 `.done` 里追加记账。
5. 单个分片失败会指数退避重试（最多 6 次）；40 个进程的重试带随机抖动，
   避免同时重试打爆端点。
6. 全部完成后按 sha256 校验，通过则写 `.complete`，并清掉该文件的 `.done`。

**关于 hf-mirror 的 Range 行为**：实测 hf-mirror 对 **LFS 大文件**正常返回 206
（分片下载有效），但对**小的非 LFS 文件**（tokenizer.json / vocab.json /
merges.txt / .gitattributes）会**忽略 Range 直接回 200 + 整个文件**，而且这类
文件是直接从镜像站本体出的，速度远低于走 CDN 的大文件。

这不影响正确性：worker 见到「offset==0 的 200」时读满 `length` 即可
（单分片文件 `length` 就等于文件大小）；见到「offset>0 的 200」则会丢弃前缀后
取所需区间，内存始终有界，不会把整个文件读进内存。代价只是这几个小文件偏慢。

## 常见问题

**下载报 `Errno 99 Cannot assign requested address`**
直连 huggingface.co 不通（IPv6 无路由）。用 `--endpoint mirror`，
或确认 `http_proxy` / `https_proxy` 可用。

**40 个进程还是慢**
瓶颈通常在出口带宽或代理，而不是并发数。实测本机聚合带宽在
10 MB/s 量级就封顶了，加进程数不会更快。可以换 `--endpoint` 试试另一条路。

**CUDA OOM**
调小 `--max-pixels` 或 `--max-new-tokens`，减少图片数量；
也可以 `--dtype float16`。

**`output_attentions=True` 报错**
需要 eager attention。`forward` 子命令默认就是 eager；
若手动构造 `Qwen3VLInference`，要传 `attn_implementation="eager"`。

**模型路径不对**
所有子命令都支持 `--model-dir`，默认是脚本同级的 `model/`。
