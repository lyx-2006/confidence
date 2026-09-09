# Phase 1 Prompt 构念与内部表征迁移验证

本目录只研究同一固定答案在不同 Phase 1 表达形式下的 source attribution（SA）是否具有行为、可读出表征、几何方向和因果干预四个层面的稳定性。230 条 trajectory audit 是 **frozen held-out diagnostic set**，不是从未查看过的确认性测试集。

## 运行

```bash
python -m dp_SA.prompt_check.run_pipeline --templates T1 T2 T3 --num-gpus 1 --smoke
python -m dp_SA.prompt_check.run_pipeline --templates T1 T2 T3 --num-gpus 1 --validation-cases 100 --steering-layers 9 12 15
python -m dp_SA.prompt_check.run_pipeline --templates T1 T2 T3 --num-gpus 1
```

可用参数包括 `--resume`、`--stages behavior probe geometry steering analyze`、`--steering-layers` 和 `--alphas -2 0 2`。当前机器只有一张 RTX 4090；双 GPU 模式采用稳定 case 分片、worker 独占日志和确定性原子合并。

`--validation-cases 100` 是独立的小规模验证模式，输出写入 `output/validation_100/`，不会冒充全量正式结果。behavior/probe 使用按 family 均衡的100条audit；steering按answer和confirmatory/exploratory比例抽100条；geometry在fold 0覆盖全部eligible answer和两侧、使用不超过100个唯一candidate的最大完整cell层级。

## 表和图回答的问题

- `behavior_pairwise_correlations.csv`：四个模板的六组行为比较。T0–T1/T2/T3 是确认性比较，其余为补充比较。除相关外还报告 MAE、整体偏移、回归斜率/截距和 canonical 方向一致率。
- `behavior_template_distributions.csv`：各模板 canonical SA 的整体位置、离散程度、方向和 hard-label 分布，帮助识别“相关很高但整体平移”的情况。
- `behavior_case_level.csv`：逐 case 的 canonical soft/hard SA及原始 `answer_matches_text/image`。这里不构造 `side`。
- `behavior_pairwise_scatter.png`：T0 与三个变体的逐 case 行为关系。
- `t0_probe_transfer_metrics.csv`：同一冻结 T0 probe 的两种目标。`tx_sa` 检验能否读取新模板实际报告，`t0_sa` 检验新模板 hidden 是否仍保留 T0 坐标；T3另有 `t3_hard_score`。
- `t0_probe_transfer_predictions.csv`：所有 probe 预测的逐 case 明细。
- `t0_probe_transfer_r2.png`、`t0_probe_transfer_correlations.png`：展示层和位置上的迁移轨迹。负 R² 是有效结果，不是运行失败。
- `vector_geometry.csv`：T1–T3独立重建方向与T0方向的signed cosine、absolute cosine和norm ratio。每模板实际capture 1,625个唯一case；8,792是复用hidden后的逻辑cells。
- `vector_cosine_by_layer.png`：不同模板方向的层级几何一致性。
- `steering_transfer_effects.csv`：各模板、层、alpha的 canonical ΔSA、hard change和natural projection剂量审计。
- `steering_transfer_vs_t0.csv`：主结果是 `S²_Tx-S²_T0`；retention ratio只有在T0分母CI不跨0且至少97.5% bootstrap分母同号时报告。
- `steering_transfer_by_template.png`：T0冻结向量对T1–T3的因果迁移。

## Side 与方向的限制

Geometry 的 `sa_side` 和 steering 的 `test_side` 都是历史 T0 verbal-SA hard class形成的冻结分组标签，不是 Real SA，也不是答案“客观来自”文本或图像。它们不得从T1–T3重新估计。T2只在评分层反转数字映射，hidden方向仍固定为 `high_image-high_text`。

## 解释规则

1. 行为迁移稳定：换表达后模型仍报告相近 attribution。
2. Probe迁移稳定：T0线性readout仍能读取新模板hidden；双目标结果用于区分行为变化和坐标变化。
3. Geometry稳定：按同一冻结T0分组独立构造出的早期方向相近。
4. Steering稳定：不改变数值或norm的T0向量能因果改变新模板输出。

只有四层共同稳定，才支持“SA不只绑定于T0固定措辞和数字顺序”。仅行为稳定不能说明内部机制稳定；probe/geometry稳定而行为不稳定，说明内部信号仍存在但最终报告规则受模板影响。LAT/PANL的迁移比CLE/SAC更能说明早期共享状态，后两者可能包含普通输出格式准备。

T3标签在冻结tokenizer下并非等长：`BALANCED`为2 tokens，其余为3 tokens。完整序列log-likelihood是预注册主定义，length-normalized结果是必要敏感性分析。如果两者的主要结论不同，应判定T3稳定性证据不足，不得择优宣称迁移稳定。

正式评分始终使用五个固定候选的完整 teacher-forcing 序列似然，不使用自由生成。仅 smoke 对少量 T3 case 运行非约束 greedy 格式审计；其 `greedy_parse_status` 只描述格式遵从，不进入行为、probe、geometry 或 steering 指标。`artifacts/diagnostics/output_validation.json` 记录表结构、case/trial 完整性、bootstrap 次数和非空图件的最终验收结果。
