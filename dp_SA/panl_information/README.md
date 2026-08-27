# delayed-SA PANL information

本模块测量 text-only / image-only restricted-candidate entropy difficulty，复放冻结的 delayed-SA Phase 1 hidden，并训练 difficulty 与 decision-side OOF probes。Probe 结果只表示信息可解码，不构成因果使用证据。

预运行（CPU tests + 2-item GPU smoke，不启动正式实验）：

```bash
python -m dp_SA.panl_information.run_pipeline
```

Smoke 成功后会打印唯一正式命令：

```bash
python -m dp_SA.panl_information.run_pipeline --formal --resume
```

各阶段也提供独立 CLI：`score_unimodal`、`capture`、`train_difficulty_probes`、`train_decision_probe` 和 `analyze`。正式输出固定在 `dp_SA/panl_information/output/results/`；默认拒绝覆盖，只有 fingerprint 完全一致时允许 `--resume`。
