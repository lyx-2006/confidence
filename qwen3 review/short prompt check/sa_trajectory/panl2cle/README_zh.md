# Short PANL→下一层CLE四格实验

该实验只使用short capture、short steering construction和short 80-case测试集。

层配对固定为PANL L14→CLE L15、PANL L16→CLE L17、PANL L18→CLE L19，alpha为-5和+5。四格为：C0 clean、C1 PANL steering自然传播、C2 steering后恢复clean CLE hidden、C3在clean PANL条件下移植steered CLE hidden。

```bash
export PYTHONDONTWRITEBYTECODE=1
python "qwen3 review/short prompt check/sa_trajectory/panl2cle/run_pipeline.py" --stage prepare
OMP_NUM_THREADS=1 python "qwen3 review/short prompt check/sa_trajectory/panl2cle/run_pipeline.py" --stage train-probes --resume
python "qwen3 review/short prompt check/sa_trajectory/panl2cle/run_pipeline.py" --stage all --smoke --resume
python "qwen3 review/short prompt check/sa_trajectory/panl2cle/run_pipeline.py" --stage run --run-formal --resume
python "qwen3 review/short prompt check/sa_trajectory/panl2cle/run_pipeline.py" --stage analyze --resume
```

`artifacts/logical_four_cell.jsonl`逐case保留四种条件的绝对final SA、CLE probe SA及相对C0变化。`tables/condition_summary.csv`同时给出绝对值与delta；`tables/effect_summary.csv`报告total、residual、attenuation、transfer和interaction。主口径为answer-equal macro，image/text侧统计只作补充，因为short测试集为74/6。
