# Short ↔ Reverse CLE Hidden Swap

本实验在相同case、相同CLE token、相同decoder block output层执行双向hidden替换。测试集按普通short canonical soft-SA选择25条text-side和25条image-side，优先远离0.5且item全局不重复。

- `reverse_to_short`：reverse clean CLE注入short，按short评分表计算输出。
- `short_to_reverse`：short clean CLE注入reverse，按reverse倒序评分表计算输出。

正式运行时重新捕获bf16 source hidden，不读取历史float16 hidden作为swap源。层为L12、L16、L18、L20、L22、L24、L26、L28、L30。

```bash
python -m cle_swap.run_pipeline --stage prepare
python -m cle_swap.run_pipeline --stage smoke --resume
python -m cle_swap.run_pipeline --stage run --run-formal --resume
python -m cle_swap.run_pipeline --stage analyze --resume
```

`delta_sa`始终是swap后减目标prompt clean的canonical image-side SA。`logit_change_diff`沿用既有clean获胜raw class margin定义，正数表示该margin被削弱。语义检验主要看`semantic_movement`是否为正、`semantic_distance_change`是否为负，并与raw-label方向指标比较。

## 同prompt、答案匹配小样本

`run_within_short_smoke.py`排除跨prompt坐标系问题，在普通short内部选择black、white、red、yellow、purple五个答案匹配对。每对由一个远离0.5的text-side case和一个image-side case组成，10个item均不同。在九层分别执行image CLE→text case和text CLE→image case，第一对另做全部18个self-swap门禁。

结果位于`output/cle_swap/within_short_answer_matched_smoke/`：`trials.jsonl`保存90条逐case swap，`summary.csv`保存逐层均值和小样本bootstrap区间，`within_short_swap.png`展示ΔSA、朝donor语义的移动及token change rate。该实验只有5对，区间只用于描述，不作为正式总体推断。

正式答案匹配运行位于`output/cle_swap/within_short_answer_matched_formal/`，覆盖全部11个可匹配答案类别、22个不同item、198条双向swap和238次forward。
