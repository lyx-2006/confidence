# Qwen3 short prompt 机制复现实验

本目录只保存short-prompt新增代码和结果。原有`qwen3 review/capture`、`Steering`、`AttentionBlock`均作为只读输入。Phase 0答案冻结自long capture；Phase 1使用给定short模板重新前向和capture。

## 运行顺序

```bash
export PYTHONDONTWRITEBYTECODE=1
python "qwen3 review/short prompt check/capture/integrity.py" snapshot
python "qwen3 review/short prompt check/capture/run.py" --preflight-only
python "qwen3 review/short prompt check/capture/run.py" --resume
python "qwen3 review/short prompt check/capture/analyze.py"
# 反向类别方向版本（0=高图像贡献，8=高文本贡献）
python "qwen3 review/short prompt check/capture_reverse/run.py" --preflight-only
python "qwen3 review/short prompt check/capture_reverse/run.py" --resume
python "qwen3 review/short prompt check/capture_reverse/analyze.py"
python "qwen3 review/short prompt check/steering/run.py"
python "qwen3 review/short prompt check/attention_block/run.py"
# 有GPU时单独运行SAC读取PANL且不重归一化的新实验：
python "qwen3 review/short prompt check/attention_block/run_sac2panl_preserve.py"
python "qwen3 review/short prompt check/capture/integrity.py" verify
```

中断后的GPU阶段使用`--resume`。位置预检单独运行后，正式capture也要带`--resume`，因为预检产物已经存在。

## Capture表和诊断

- `output/capture/position_preflight.jsonl`：逐case列出LAT、PANL、CLE、SAC的rendered目标字符范围、token覆盖字符范围、processed index、token ID和decoded token，回答“short prompt上的语义位置是否真的对齐到预期processed token”。
- `processor_audit.json`：记录实际Fast tokenizer与Fast image processor配置，回答“运行时究竟用了哪套processor”。
- `results.jsonl`：500条short clean九类logit、概率、soft/hard-SA及hidden索引，是后续所有实验的基线。
- `output/comparison/tables/clean_case_level.csv`：每条long/short配对结果。
- `clean_long_short_metrics.csv`与`clean_soft_sa_scatter.png`：回答prompt压缩后clean输出是否仍接近，包含Pearson、Spearman、MAE和相对0.5的方向一致率。

CLE严格先唯一定位完整`Source attribution classes:\n0,1,2,3,4,5,6,7,8`，再取末尾首换行所在processed token。当前该token同时包含两个换行，decoded为`\n\n`；这是人工确认的“包含目标换行token”定义，不是邻位猜测。

## Steering表和图

- `selection_summary.json`：回答依据short clean结果实际选出了哪些25/25 construction，以及排除construction item后最靠近soft-SA 0.5的80个item-disjoint测试case。`test_side`只按soft-SA低于/高于0.5作描述，不是配额条件。
- `short_vector_metadata.json`：记录short独立方向的raw norm、short residual norm和3%缩放。
- `predictions.jsonl`：short重建方向与long→short迁移的全部逐case干预结果。
- `tables/alpha_effects.csv`：每个模式、位置、层、alpha的平均Δsoft-SA、hard-midpoint变化、hard-class change rate及family-bootstrap CI。
- `tables/symmetric_effects.csv`：回答正负剂量是否产生方向相反的对称效应；`symmetric_soft_sa=(Δ+a-Δ-a)/2`，`asymmetry_soft_sa=Δ+a+Δ-a`。
- `tables/parity_summary.json`：回答alpha=0是否与clean完全一致、hook是否仅应用一次。
- `figures/short_rebuilt_dose_response.png`：short独立重建方向的逐层剂量效应。
- `figures/long_vector_to_short_dose_response.png`：long方向能否迁移到short prompt。
- `figures/symmetric_effect_comparison.png`：直接比较两种方向的S5对称效应。

## Attention Block表和图

- `shared_eager_gate/assessments.jsonl`、`excluded.jsonl`和`summary.json`：回答原100个候选中哪些short case满足SDPA↔eager clean门禁；失败case不补选。
- 每个实验的`artifacts/trials.jsonl`：逐case clean、main block和source+1 control结果及attention审计。
- `tables/condition_effects.csv`：main/control各自的signed/absolute Δsoft-SA、token change rate和logit change diff。
- `tables/paired_main_vs_control.csv`：主要证据，回答main是否比相邻source+1 control产生更强配对效应。
- `tables/long_short_paired_contrasts.csv`：在同一门禁交集上报告long、short的main-control点估计及paired short-minus-long CI。
- `figures/short_main_control_effects.png`：short main与control自身效应。
- `figures/long_short_main_control_contrast.png`：long/short的main-control配对contrast。

## Short SA trajectory 四格

`sa_trajectory/panl2cle/`复用原SA trajectory的四格hook与probe方法，但只使用short数据：short Steering的80条测试case、short 25/25 construction方向，以及short训练的CLE L15/L17/L19 probe。配对为PANL L14→CLE L15、L16→L17、L18→L19，alpha为±5。

- `output/sa_trajectory/PANL2CLE_four_cell/artifacts/logical_four_cell.jsonl`：逐case保留C0 clean、C1自然传播、C2恢复clean CLE、C3移植steered CLE四种条件的final SA和CLE probe SA。
- `tables/condition_summary.csv`：四种条件的绝对SA、相对C0 delta及95% CI；`tables/effect_summary.csv`：total/residual/attenuation/transfer/interaction分解。
- `figures/four_cell_final_soft_sa.png`和`four_cell_cle_probe_sa.png`：回答Steering后最终SA与CLE probe SA是否同步变化；`figures/effects_final_and_probe.png`：比较四格效应分解。
- 物理网格为1520 trials，展开后1920条逻辑行；正式运行前必须通过probe可靠性与alpha-zero gate。

所有CI使用seed 42、2000次family bootstrap。Attention主口径是answer-equal macro，同时保留family-micro及两侧结果。

### SAC2PANL_preserve新增实验

- query固定为`P1_SAC`；main source为`P1_PANL`，control为`P1_PANL_PLUS_1`。
- 层窗口为闭区间L8–12、L12–16、L16–20、L20–24；使用现有gate通过的64个共同case。
- 该实验与前两项pre-softmax mask不同：它在普通eager softmax完成后只把指定边置零，不重新归一化。因而被阻断query row的行和会减少，其他attention矩阵元素必须逐项、bitwise不变。
- 每层诊断保存`max_removed_weight`、`max_blocked_weight_after`、`max_other_weight_change`、行和质量守恒误差和hook次数；任一非目标权重变化即停止。
- 指标为`delta_soft_sa`、`logit_change_diff`和`token_change_rate`；分别输出main/control自身统计及逐case main减PANL+1 control的family-bootstrap CI。
- 无GPU代码检查：`python "qwen3 review/short prompt check/attention_block/run_sac2panl_preserve.py" --check-only`。
- GPU smoke使用一条text-side和一条image-side case，并覆盖全部四个层窗口，结果隔离在`SAC2PANL_preserve/smoke/`：`python "qwen3 review/short prompt check/attention_block/run_sac2panl_preserve.py" --smoke`。

## 能够和不能够支持的结论

- clean接近且short仍有相同传播效应：支持该现象不依赖详细九级类别描述。
- clean接近但传播效应改变：说明相似行为输出可能由不同内部路径实现。
- clean本身明显改变：只能说明prompt压缩改变了任务，不能用传播差异否定原机制。
- attention block显著：只说明指定attention读取具有功能作用，不能声称它是唯一传播路径。
- long vector在short上有效说明方向具有跨prompt迁移性；它不等同于short独立重建方向，也不证明两种prompt内部表征完全相同。

## Short Reverse Prompt

`capture_reverse/` 使用相同500个case和冻结的Phase 0答案，重新渲染反向类别说明并重新保存完整140个hidden向量。原始模型标签仍按0–8保存，但 canonical midpoint倒序，因此 `soft_sa_image_score` 仍与普通short保持“越高越偏图像侧”的统一方向。

- `output/capture_reverse/results.jsonl`：反向prompt的500条九类logit、概率、raw类别、canonical soft/hard-SA和完整hidden索引。
- `output/capture_reverse/tables/short_reverse_case_comparison.csv`：与普通short逐case配对，包含raw类别反转检查和canonical分数差。
- `output/capture_reverse/tables/short_reverse_metrics.csv`：Pearson、Spearman、MAE、0.5方向一致率、raw类别反转率和canonical hard-class一致率。
- `output/capture_reverse/figures/short_reverse_canonical_sa_scatter.png`：回答反转prompt在统一canonical SA尺度上是否复现普通short的clean评分。

反向类别映射是评分定义，不会强制模型输出类别满足 `reverse_raw = 8 - short_raw`；该等式的实际满足率单独报告。

## Short ↔ Reverse CLE Hidden Swap

`cle_swap/` 在50个普通short极端case（text-side 25、image-side 25，item不重复）上，逐层交换普通short与reverse prompt的CLE hidden。正式层为L12、L16、L18、L20、L22、L24、L26、L28、L30；每个case分别重捕获两套prompt的bf16 clean CLE，再执行`reverse→short`和`short→reverse`，不把历史float16 hidden直接作为swap源。

- `output/cle_swap/artifacts/manifests/test_manifest.jsonl`：50条配对测试样本及两套clean SA，侧别按普通short定义。
- `output/cle_swap/artifacts/trials/`：100条clean记录与900条逐层双向swap记录；`artifacts/trials.jsonl`只汇总cross-swap trial。
- `tables/direction_layer_metrics.csv`：每个方向×层的ΔSA、absolute ΔSA、logit change diff、token/hard-class变化率、语义距离和raw-label距离指标及2000次item-bootstrap CI。
- `tables/bidirectional_paired_metrics.csv`：两个方向是否同时向各自source语义移动。
- `tables/gap_associations.csv`：ΔSA与source-target canonical gap及raw-label gap的相关/斜率。
- `figures/delta_sa_by_layer.png`、`logit_and_token_change.png`、`semantic_vs_raw_attraction.png`、`bidirectional_paired_effects.png`：分别回答CLE swap是否改变最终SA、logit/token是否变化、变化更接近语义source还是raw标签source，以及双向结果是否一致。

Self-swap smoke门禁覆盖两个case和全部9层；正式运行共1000次forward（100 clean + 900 cross-swap）。
