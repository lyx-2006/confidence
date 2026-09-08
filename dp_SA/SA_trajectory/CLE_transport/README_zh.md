# CLE attention transport blocking

本目录包含两个窗口级 attention transport blocking 实验。它们复用项目已有的完整 delayed-SA prompt、Fast processor、multimodal input、位置定位、九类 soft SA 评分和 eager-attention mask hook。

## 实验

- `PANL2CLE`：让 `P1_CLASS_LIST_END`（CLE）不能直接读取 `P1_PANL`；相邻控制阻断 `P1_PANL_PLUS_1`。
- `CLE2SAC`：让 `P1_SAC` 不能直接读取 `P1_CLASS_LIST_END`；相邻控制阻断其 processed index `+1`。
- 四个窗口分别为 L8–12、L13–17、L18–22、L23–26。窗口内所有层和所有 attention heads 同时阻断同一条 query→source edge，因此不能解释为单层效应。

阻断只把指定 pre-softmax mask 坐标设为 dtype 最小值。被阻断 edge 的 post-softmax 权重必须为 0，attention row 必须重新归一化；同一 row 的其他权重随重归一化而变化是正常现象。

## 运行

先分别执行 smoke：

```bash
python -m dp_SA.SA_trajectory.CLE_transport.run_pipeline --experiment PANL2CLE --smoke
python -m dp_SA.SA_trajectory.CLE_transport.run_pipeline --experiment CLE2SAC --smoke
```

smoke 通过后运行正式实验：

```bash
python -m dp_SA.SA_trajectory.CLE_transport.run_pipeline --experiment PANL2CLE
python -m dp_SA.SA_trajectory.CLE_transport.run_pipeline --experiment CLE2SAC
```

支持 `--windows 8-12,18-22`、`--num-gpus 1|2`、`--resume` 和 `--output-root PATH`。分批补窗口时，对同一输出目录使用 `--resume`；已经完成的 forward 不会重复执行。

## 指标与解释

- `delta_soft_sa = blocked_soft_sa - clean_soft_sa`；正值更偏图像侧。
- `token_change_rate` 表示 hard SA class 相对 clean 是否改变。
- `logit_change_diff = clean_margin - blocked_margin`；margin 是 clean hard class logit 减去其余八类 logit 的均值，正值表示原决策被削弱。
- `specific_effect = main_block_effect - source_plus_1_control_effect`。

主要统计使用 answer-equal macro 和 2000 次 family bootstrap。正结果只能说明该直接 edge 在对应窗口具有功能作用；负结果不能证明信息路径不存在，因为信息可能已在更早层被复制或经其他路径传递。
