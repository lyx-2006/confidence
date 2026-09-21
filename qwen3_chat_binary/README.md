# Qwen3 多轮五分类来源归因 Steering

该目录实现一个与旧实验隔离的 Qwen3-VL-8B-Instruct 实验。数据来自
`generate dataset/datasets/current/conflict_test.json` 的 396 个独立 conflict case。
第一阶段使用原始
`dp_SA` answer prompt 生成一次回答；第二阶段把该回答作为真实 assistant 历史，追加
五分类来源归因问题，并在固定回答的条件下对历史位置进行 activation steering。
第二阶段 prompt 采用已选定的五分类版本，与 `prompt test` 中保存的
`v28_fiveway_rule_before_labels` 文本完全一致。

实验比较两个历史边界版本：

| 版本 | PANL | PANL+1 |
|---|---|---|
| `native_boundary` | 历史 assistant 的 `<|im_end|>` 后换行 | 下一条 user 的 `<|im_start|>` |
| `explicit_newline` | 人工追加到历史回答末尾的换行 | 历史 assistant 的 `<|im_end|>` |

两版共享第一阶段回答、construction/test manifest，但分别捕获 hidden state、构造
`mean(high_image) - mean(high_text)` 向量。向量归一化为构造样本平均 residual norm
的 3%。归因指标是在 `**Source Attribution**:` 后对单 token `0`–`4` logits 做
五项 softmax，再用权重 `0, .25, .5, .75, 1` 计算期望位置。该值是
`image_attribution_score`；对应的有符号分数为 `2*score-1`。这些指标只描述事后归因
报告倾向。

## 目录

```text
qwen3_chat_binary/
├── capture.py             # 共享 Phase 0 + 两版 Phase 1 hidden capture
├── steering.py            # 共享划分、分版向量和参数化干预
├── conversation.py        # 四消息对话与 assistant prefill
├── dataset.py             # conflict_test case 校验与路径解析
├── positions.py           # LAT/PANL/PANL+1/CLE/SAC 动态定位
├── scoring.py             # 0–4 单 token 概率与期望位置映射
├── smoke.py               # 串联 smoke；全程成功后自动删除临时输出
├── summarize_results.py   # 总体、分侧和配对汇总
├── plot_results.py        # 两版分位置层间对照图
├── previous_results/      # 历史实验结果归档
├── output/
│   ├── Capture/
│   │   ├── progress|figures|tables/
│   │   └── <variant>/progress|figures|tables/
│   └── Steering/
│       ├── progress|figures|tables/
│       └── <variant>/progress|figures|tables/
└── tests/
```

大体积 capture 和 output 由本目录 `.gitignore` 排除。

## 验证与运行

CPU 和 processor 测试：

```bash
python -m pytest -q qwen3_chat_binary/tests
```

正式 capture 前先运行 20-case capture smoke。成功后，
`output/_smoke/run-*` 临时目录会自动删除；失败时保留，以便检查诊断：

```bash
python -m qwen3_chat_binary.smoke --capture-only
```

不带 `--capture-only` 时会继续执行精简 steering smoke。

正式 capture 默认按 JSON 顺序处理全部 396 条 case，并捕获 layer 8–35：

```bash
setsid nohup python -m qwen3_chat_binary.capture \
  > qwen3_chat_binary/output/Capture/progress/run.log 2>&1 &
# 中断后继续时必须保持配置完全相同
python -m qwen3_chat_binary.capture --resume
```

正式 steering 默认只使用 `LAT PANL CLE` 三个位置、layer `8 12 18 24 30 35` 和 alpha
`-5 -2 0 2 5`：

```bash
setsid nohup python -m qwen3_chat_binary.steering \
  > qwen3_chat_binary/output/Steering/progress/run.log 2>&1 &
python -m qwen3_chat_binary.steering --resume
```

生成汇总和对照图：

```bash
python -m qwen3_chat_binary.summarize_results
python -m qwen3_chat_binary.plot_results
```

每个 output root 绑定模型、数据、prompt、边界定义和参数网格指纹。改变任一配置时
必须使用新的 output root，不能跨配置 resume。

## 输出与检查

- `output/Capture/tables/phase0_results.jsonl`：两个版本共享的实际第一阶段输出和生成 token。
- `output/Capture/<variant>/tables/results.jsonl`：clean 五分类分数、动态位置和诊断。
- `output/Capture/<variant>/tables/hidden/*.npz`：五位置乘 28 层的 float16 hidden states。
- `output/Steering/tables/{construction,test}_manifest.jsonl`：按 native clean 分数建立的共享 case 划分。
- `output/Steering/<variant>/tables/vectors.pt`：该版本独立计算的方向。
- `output/Steering/<variant>/tables/predictions.jsonl`：逐样本干预结果、五类概率质量、扰动范数和 hook 诊断。
- `output/Steering/figures/` 与各版本的 `figures/`：跨版本与分版本图。

旧二分类主实验结果完整归档在 `previous_results/binary_chat_template/`，不会被新实验的
默认入口读取。

## 反事实行为依赖

`prepare_counterfactual` 使用当前 396-case 数据集，重建原图并改色，随后从当前文本池
选择同难度的第三色线索。`run_counterfactual` 对成功构造的 case 运行原图/改图 × 原文/替换文
四个条件，保存概率版和 logit 版 CMA；`analyze_counterfactual` 生成 CMA–SA 统计与散点图。

```bash
setsid nohup python -m qwen3_chat_binary.prepare_counterfactual \
  > qwen3_chat_binary/output/Counterfactual/progress/prepare.log 2>&1 &
python -m qwen3_chat_binary.prepare_counterfactual --resume
setsid nohup python -m qwen3_chat_binary.run_counterfactual \
  > qwen3_chat_binary/output/Counterfactual/progress/run.log 2>&1 &
python -m qwen3_chat_binary.analyze_counterfactual
```

构造失败的 case 会进入 `construction_manifest.jsonl` 并保留候选尝试；四条件测量只消费
构造成功的 case，不自动放宽 entropy 或目标概率匹配阈值。

## Answer-matched Steering

`prepare_answer_matched` 将 native-boundary 的 396 条 capture 固定分为 317 条构造和
79 条测试，并按第一阶段答案颜色构造 LOAO SA 方向。实验只运行 `matched_loao`，默认覆盖
LAT/PANL/CLE、层 8/12/18/24/30/35 和 alpha -5/-2/0/2/5。

```bash
python -m qwen3_chat_binary.experiments.answer_matched_steering.pipeline --smoke-only
python -m qwen3_chat_binary.experiments.answer_matched_steering.prepare
setsid nohup python -m qwen3_chat_binary.experiments.answer_matched_steering.run \
  > qwen3_chat_binary/output/AnswerMatchedSteering/native_boundary/progress/run.log 2>&1 &
python -m qwen3_chat_binary.experiments.answer_matched_steering.analyze
```

正式实验产生 7,110 条干预记录。成功 smoke 自动删除临时目录；失败 smoke 保留诊断。

程序强制检查 `alpha=0` parity、hook 只注入一次，以及 layer 35 在所有非 SAC
历史位置上的传播负对照。失败样本会显式写入 capture 结果；选择阶段只使用两个版本
均成功且连续期望分数不等于 0.5 的 case。construction 与 test 按 `case_id` 去重且
互不重叠。正式实验按连续分数小于或大于 0.5 划为 `text_side`、`image_side`，要求两侧都非空；所有选择和去重
均只基于 `case_id`，smoke 只要求 10 条 case-disjoint 测试样本。
