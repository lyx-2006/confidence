# Short Reverse Prompt Capture

本目录实现 short prompt 的方向反转版本。除以下一句外，模板文字、换行和段落布局与 short capture 完全相同：

- reverse prompt：高类别表示更强文本贡献，低类别表示更强图像贡献。

模型输出的原始类别仍按 `0..8` 保存。评分表使用原 canonical midpoint 的倒序：

`[0.95, 0.825, 0.675, 0.5625, 0.5, 0.4375, 0.325, 0.175, 0.05]`

因此：

- `raw_argmax_class` 是模型按 reverse prompt 输出的原始类别；
- `argmax_hard_class = 8 - raw_argmax_class`，转换回高值代表图像侧的统一方向；
- `soft_sa_image_score` 也保持高值代表图像贡献，可与原 short 结果直接比较。

输出位于 `../output/capture_reverse/`。每个case重新前向运行并保存 LAT、PANL、CLE、PANL+1、SAC 在L8–L35的140个float16 hidden向量，不复用原short hidden。

运行：

```bash
python capture_reverse/run.py --preflight-only
python capture_reverse/run.py --smoke 2
python capture_reverse/run.py --resume
python capture_reverse/analyze.py
```

比较表报告 canonical image-side soft-SA 的 Pearson、Spearman、MAE、0.5方向一致率，同时报告原始类别是否满足 `reverse_raw = 8 - short_raw`。类别反转一致并非强制门禁，因为prompt反转可能改变模型行为。
