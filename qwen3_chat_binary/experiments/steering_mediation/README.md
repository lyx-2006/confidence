# 五分类提示词 Steering Mediation

本实验仅使用 `native_boundary`，测试集为连续 SA 分数在各侧最接近 0.5 的 50+50 条；方向构造集为两侧极端 25+25 条，与测试集不重叠。两条链路是 LAT→PANL 和 PANL→CLE，层对固定为 L14→15、L16→17、L18→19，alpha 为 −5 和 +5。

每个格点运行 C0 clean、C1 steering 自然传播、C2 steering 后恢复同样本 clean 下游 hidden、C3 clean 上游下移植同样本 C1 corrupted hidden。donor 由本次 forward 原 dtype 保存。

```bash
python -m qwen3_chat_binary.experiments.steering_mediation.prepare
python -m qwen3_chat_binary.experiments.steering_mediation.pipeline --smoke
python -m qwen3_chat_binary.experiments.steering_mediation.pipeline --resume
```

`prepare` 是 CPU-only，会生成真实 manifests、六个位置/层方向、processor audit 和运行指纹。`run` 需要 GPU；`analyze` 报告 final SA 的总效应、恢复后残余、恢复消除量和移植效应。
