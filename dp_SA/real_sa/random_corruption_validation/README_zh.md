# Random Corruption Intervention Robustness

本目录实现“随机破坏干预稳健性”实验，是对 `dp_SA/real_sa/` 中 mean-embedding 极端干预的独立替代检查。它不会修改原 Real SA 代码或历史输出，也不应被表述为新的、唯一的或更真实的 ground-truth Real SA。

实验固定使用原有 100 个测试 case（50 个 family）和五次随机重复。`v11` 保留图像与文本；`v10` 只把 Text clue 的内容 token 原位替换；`v01` 只把整张图像替换为高斯噪声 PNG；`v00` 同时采用该重复中与 `v10` 相同的随机 token、与 `v01` 相同的噪声图。随机 token 来自冻结训练线索的经验频率分布。单次抽样可能偶然形成有语义的片段，因此主要结果先对五次干预取平均，并另行报告重复稳定性。

运行方式：

```bash
CUDA_VISIBLE_DEVICES=0 python -m dp_SA.real_sa.random_corruption_validation.run \
  --mode smoke --output-root dp_SA/real_sa/random_corruption_validation/output --resume

CUDA_VISIBLE_DEVICES=0 python -m dp_SA.real_sa.random_corruption_validation.run \
  --mode formal --output-root dp_SA/real_sa/random_corruption_validation/output --resume
```

Formal 必须先有相同运行指纹的 passing smoke。温度 `1.0` 仅用于 12 个固定答案候选 logits 的受限 softmax，不是 Qwen 文本生成的采样温度。正式运行先重新计算全部 clean 分数并与已有 mean-embedding Real SA 的 clean 条件严格校验；任何失败都会在 corruption 评分前终止。

主要量由已有 `case_metrics()` 定义：`D_I`、`D_T` 是破坏单一模态造成的支持度变化；`phi_I`、`phi_T` 是两模态的 Shapley 贡献；`J` 是交互项；`G_R = phi_I - phi_T`，正值表示图像贡献相对更大。输出同时比较随机干预与 mean-embedding 干预，并比较新 `G_R` 与 `2 × soft_sa_image_score - 1`。置信区间使用 seed 42、2000 次 family-cluster bootstrap，不能把 500 个随机重复视为相互独立的 case。

Smoke 结果完全位于 `output/progress/smoke/`。Formal 结果位于 `output/artifacts/`、`output/tables/` 和 `output/figures/`；代码不会自动启动 Formal。
