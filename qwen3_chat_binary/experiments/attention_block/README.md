# 五分类提示词 Attention Block

本实验仅使用 `native_boundary`，从当前 capture 中按连续 SA 分数选取文本侧最低 50 条和图像侧最高 50 条。SAC 作为 query，分别阻断到 PANL、PANL+1、CLE、CLE+1 的单条 attention edge；四组闭区间窗口为 L12–16、L16–20、L20–24、L24–28。

```bash
python -m qwen3_chat_binary.experiments.attention_block.prepare
python -m qwen3_chat_binary.experiments.attention_block.pipeline --smoke
python -m qwen3_chat_binary.experiments.attention_block.pipeline --resume
```

`prepare` 是 CPU-only，会完成真实 100 条选样、processor 位置复核和指纹冻结。`run` 需要 GPU 和 eager attention；每条阻断以同次 eager clean 为基线，记录五类 logits、token change rate、logit change diff、并列诊断及 attention audit。`analyze` 输出分侧汇总及主位置减邻位控制的配对结果。

## PANL steering 后增强阻断

增加 `--steered-block` 后使用独立输出目录，不影响上述旧版结果。增强版在 PANL L16 注入 alpha −5/+5，再在 L17–24、L22–30、L24–32、L26–34 阻断 SAC→PANL，并以相同 alpha 的仅 steering forward 为主基线：

```bash
python -m qwen3_chat_binary.experiments.attention_block.pipeline --steered-block --smoke
python -m qwen3_chat_binary.experiments.attention_block.pipeline --steered-block --resume
```

smoke 额外强制检查重复 clean、alpha=0 和空阻断的 logits 一致性。正式分析输出 SB 相对 S，以及 S/SB 相对 clean 的 token change rate 和 logit change diff。
