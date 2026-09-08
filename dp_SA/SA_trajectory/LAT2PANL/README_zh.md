# LAT→PANL SA中介验证

本目录独立验证：在 `P1_LAT` 的decoder L14注入answer-matched `matched_loao` SA方向后，效应是否经由 `P1_PANL` 的L14–L18状态传播到CLE L20与最终SAC。

四个条件为：clean LAT + clean PANL；Steering LAT + PANL；Steering LAT + clean PANL；clean LAT + Steering PANL。所有patch来源均以原始bf16位模式保存，不经过float16。

正式final soft SA使用冻结的174例。CLE frozen-probe只使用family、item、image hash均不与probe训练集重叠的73例；因brown只有1例且没有high-image，Figure 2预先标为探索性。

## 命令

CPU测试：

```bash
OMP_NUM_THREADS=1 python -m pytest -q dp_SA/SA_trajectory/LAT2PANL/tests
```

单卡smoke（24 case，360 forward）：

```bash
python -m dp_SA.SA_trajectory.LAT2PANL.run_pipeline --smoke --num-gpus 1
```

正式输入审计（不会启动正式forward）：

```bash
python -m dp_SA.SA_trajectory.LAT2PANL.run_pipeline
```

通过smoke后，显式启动正式单卡或双卡实验：

```bash
python -m dp_SA.SA_trajectory.LAT2PANL.run_pipeline --run-formal --num-gpus 1 --resume
python -m dp_SA.SA_trajectory.LAT2PANL.run_pipeline --run-formal --num-gpus 2 --resume
```

上面只有带 `--run-formal` 的命令会启动174-case正式实验。GPU数量只影响case分片，不进入语义fingerprint；可在1/2卡之间交叉resume。

主要统计量是绝对 `S_attenuation` 与paired family-bootstrap CI。ratio只有在总效应CI不跨0且至少97.5%的bootstrap分母与点估计同号时才报告，不能解释成严格自然间接效应。
