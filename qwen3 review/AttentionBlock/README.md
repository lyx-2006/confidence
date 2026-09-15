# Qwen3-VL attention-block experiments

本实验在 Qwen3-VL eager attention 中阻断一条精确的 query→source edge，并以同一
eager runtime 的 clean forward 为配对基线。测试 manifest 从 500 条 capture 中用
seed 42 重新选择：图像侧 50、文本侧 50，100 个 item 全局不重复。

- `CLE2SAC`：SAC→CLE；控制 SAC→CLE+1；闭区间 L12–16、L16–20、L20–24、L24–28。
- `PANL2CLE`：CLE→PANL；控制 CLE→PANL+1；闭区间 L8–12、L12–16、L16–20、L20–24。

运行 smoke：

```bash
python "qwen3 review/AttentionBlock/run_pipeline.py" --experiment CLE2SAC --smoke
python "qwen3 review/AttentionBlock/run_pipeline.py" --experiment PANL2CLE --smoke
```

正式运行：

```bash
python "qwen3 review/AttentionBlock/run_pipeline.py" --experiment CLE2SAC
python "qwen3 review/AttentionBlock/run_pipeline.py" --experiment PANL2CLE
```

中断后为相同命令增加 `--resume`。输出包括逐样本 trial、attention audit、窗口统计、
主阻断减邻位控制的 paired 统计，以及 delta soft-SA、token change rate、logit change
diff 三张图。

跨 attention backend 的 clean gate 要求 hard class 一致、九类 raw-logit 最大误差不超过
1.0、soft-SA 误差不超过 0.01；所有实验效应始终以同次 eager clean 为配对基线。
