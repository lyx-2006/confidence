# LAT→PANL/SAC 逐层 trajectory

独立实验包。它只在本目录的 `output/` 中写入结果；父实验的 manifests、hidden、probes、vectors 和历史 trials 都按 SHA256 只读审计。模型输入处理显式复用父实验的 `Qwen2VLImageProcessorFast` 策略，不依赖 Transformers 的默认选择。

默认入口：

```bash
python -m dp_SA.confidence_steering.trajectory.run_pipeline --num-gpus 1
```

交付验收 smoke：

```bash
python -m dp_SA.confidence_steering.trajectory.run_pipeline --smoke --num-gpus 1
```

可加 `--resume`，或用 `--output-root` 指定 `trajectory/output/` 下的子目录。`num_gpus` 仅是执行元数据，不进入语义 fingerprint。

Smoke 固定使用 audit 的四个完整 family、24 case。每 case 跑一个共享 α=0 baseline，以及 `confidence_raw`、`confidence_parallel_sa`、`confidence_perp_sa_natural_scale` 三个方向各自的 ±0.5，共 168 forward。当前单卡执行完成后，程序用这些真实结果模拟单/双 worker 分片和 canonical merge，不宣称进行了真实双卡推理。

正式流程先捕获 1112 case 的 Fast clean hidden、训练 208 个 probe，通过独立 smoke 后才读取并校验封存的 100-case formal manifest，随后运行 700 forward。正式 α=0 的历史 reference 是父实验 `output/all_fast_l14` 的 100-case explicit-Fast clean baseline；旧 Slow `natural_decomposition` trials 仅可作为 processor 稳健性对照，不能作为 Fast parity reference。所有 α=0 parity 失败都会完整记录但不会中止其余 trajectory。只有全部 probe、hidden、表格、图、中文文档、hash、parity 记录和 merge 门禁通过才产生 `completion.json`。

`confidence_perp_sa_natural_scale` 在报告中统一解释为 “SA-subspace-orthogonal confidence-related component”。所有对称导数均为 `(Y(+0.5)-Y(-0.5))/1.0`，component additivity 只解释为低剂量近似。
