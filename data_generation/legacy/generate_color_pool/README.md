# generate_color_pool.py

五档文本熵颜色池（V2 Entropy 单入口）的独立运行脚本。

## 概述

读取问题数据集（默认 `datasets/dataset.json`），对每个 `颜色 × entropy bin`
运行「DeepSeek 候选生成 → 契约校验 → DeepSeek Analyzer → 三次 Qwen 测试」，
把通过的线索写入五档先验池（默认
`generation_v2_outputs/formal/text/text_entropy_pool.json`，schema
`text_entropy_pool.v2`）。

- **熵分档**：`[0,20) [20,40) [40,60) [60,80) [80,100]`（12 类颜色受限答案
  空间的自然对数熵，除以 `ln(12)` 映射到 0–100）。
- **通过条件**：三次答案均为目标色、restricted top-1 全是目标色、三次实测
  熵落在同一档、且 `max-min` 小于该档容差（`5 + 2.5 × bin_id`：
  5.0 / 7.5 / 10 / 12.5 / 15）。允许跨档路由（实测档归档，并按实测档容差
  复核）。
- 与图像入口共用同一个 Qwen batch 运行时（`generation_runtime.py`：父进程
  唯一模型实例、逐 batch 串行）；Qwen 测试队列位于按 run root 哈希命名的
  系统临时目录，不落输出目录。

## 参数表格

| 参数 | 类型 | 默认值 | 说明 | V2 是否生效 |
| --- | --- | --- | --- | --- |
| `--find` | flag | 关 | 已废弃；V2 总是先验证再入队。 | 否（兼容保留） |
| `--after COLOR` | str | 无 | 从 12 色顺序中该颜色的下一个颜色开始。 | ✅ |
| `--round N` | int | `5` | 每个 (颜色, bin) 最多执行的 Generator/Analyzer 轮数；必须为正。 | ✅ |
| `--input PATH` | str | `datasets/dataset.json` | 输入问题数据集。 | ✅ |
| `--output PATH` | str | `generation_v2_outputs/formal/text/text_entropy_pool.json` | 主结果 JSON。 | ✅ |
| `--target-per-bin N` | int | `5` | 每个颜色、每个 entropy bin 至少保留的 prior 数量；必须为正。 | ✅ |
| `--select_pool BINS` / `--select-pool BINS` | str | `all` | 需要生成的熵档；支持 `0,1,4`、`bin0`、`40-80` 等写法（见下方「档位选择写法」）。 | ✅ |
| `--bin-batch-sizes A,B,C,D,E` | 5 个 int | `20,20,20,20,20` | Bin 0–4 每轮每档 DeepSeek 候选数；必须恰好 5 个正整数。 | ✅ |
| `--deepseek-workers N` | int | `27` | 已失效（V2 按 bin 并行调用 DeepSeek，不再使用进程池）。 | 否（仅校验 > 0） |
| `--color-workers N` | int | `6` | 已失效（V2 每颜色/bin 独立轮次，不再并发颜色）；仍校验 ≤ 6。 | 否（仅校验 > 0 且 ≤ 6） |
| `--colors A,B,C` | str | 全部 12 色 | 只处理逗号分隔的颜色子集；非法颜色名报错。 | ✅ |
| `--resume` | flag | 关 | 显式启用恢复语义；formal 输出非空时必需。 | ✅ |
| `--seed N` | int | `42` | 问题选择、模板改写等确定性随机种子。 | ✅ |
| `--near-duplicate-threshold X` | float | `0.88` | 近重复文本过滤阈值（0–1，越高越严格）。 | ✅ |
| `--stability-threshold X` | float | `0.1` | 已失效（V2 用 entropy 档位 + 容差判定稳定性）。 | 否（仅校验 > 0） |
| `--model-path PATH` | str | `qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct` | 本地 Qwen 权重目录。 | ✅ |
| `--api-config-path PATH` | str | `api_config.json` | DeepSeek API 配置 `{"api_key": ..., "base_url": ...}`。 | ✅ |
| `--qwen-batch-size N` | int | `4` | 单次 Qwen batch 的测试 job 数（1–64；24 GB 卡建议保持 4）。 | ✅ |
| `--qwen-batch-wait-ms N` | int | `500` | scheduler 攒 batch 的最大等待毫秒数（非负）。 | ✅ |
| `--qwen-wait-timeout X` | float | `86400.0` | 等待单个测试 job 完成的最长秒数（必须为正）。 | ✅ |

### 档位选择写法（`--select_pool`）

| 写法 | 含义 |
| --- | --- |
| `all` | 全部 5 档（默认） |
| `0,1,4` | 只生成最低两档和最高档 |
| `40-80` | 等价 score 区间写法（覆盖档 1、2、3） |
| `bin0` / `0.0-0.2` / `0-20` 等 | 兼容别名（旧 confidence 0–1 区间也接受） |

区间必须对齐 `0,20,40,60,80,100` 边界，否则报错。

## 校验规则（parse_args）

- `--round`、`--target-per-bin`、`--deepseek-workers`、`--color-workers`
  必须为正；`--color-workers` 不能超过 6。
- `--near-duplicate-threshold` 必须在 0–1；`--stability-threshold` 必须为正。
- `--qwen-batch-size` 必须在 1–64；`--qwen-batch-wait-ms` 非负；
  `--qwen-wait-timeout` 为正。
- `--after` 和 `--colors` 同时应用后无剩余颜色时报错。
- 正式输出根目录非空且未传 `--resume` 时报错，要求显式 `--resume`。
- **Resume 配置校验**：除 `--bin-batch-sizes` 外，其余配置须与 pool 内
  saved config 全等（batch 大小只影响每轮候选数，不影响已入库 prior 语义，
  故豁免）。

## 常用命令

```bash
# 完整运行（12 色 × 5 档）
python "data_generation/legacy/generate_color_pool/generate_color_pool.py"

# 小规模试运行：单色、单轮、每档 1 条
python "data_generation/legacy/generate_color_pool/generate_color_pool.py" \
  --colors red --round 1 --target-per-bin 1

# 指定颜色与档位
python "data_generation/legacy/generate_color_pool/generate_color_pool.py" --colors red,blue
python "data_generation/legacy/generate_color_pool/generate_color_pool.py" --select_pool 0,1,4
python "data_generation/legacy/generate_color_pool/generate_color_pool.py" --select_pool 40-80

# 从指定颜色之后开始（跳过之前的颜色）
python "data_generation/legacy/generate_color_pool/generate_color_pool.py" --after yellow

# 恢复运行（formal 输出非空时必需）
python "data_generation/legacy/generate_color_pool/generate_color_pool.py" --colors yellow --resume
```

## 注意事项

- DeepSeek 超时由 `generate_shape_color_dataset.py` 的 `DeepSeekAgents`
  统一控制（当前 `timeout=150s`、`max_retries=0`，由 producer 自身三次重试）。
  服务端高峰期单请求可达 100s+，连续超时多为服务端慢，可先跑
  `test_deepseek_connection.py` 诊断。
- 每档 DeepSeek prompt 在
  `data_generation/prompts/text_entropy_bin_prompts.json`（schema
  `text_entropy_bin_prompts.v1`），各档 Generator/Analyzer 独立特异化，
  可直接编辑 JSON 调优，无需改代码。
- 旧 confidence `PoolBuilder` 拆到同目录 `legacy_pool_builder.py`；
  `from generate_color_pool import PoolBuilder` 仍可用，但 CLI 不再调用它。
