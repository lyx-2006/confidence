# Qwen3-VL PANL → CLE four-cell mediation

This experiment trains frozen linear readouts at CLE decoder layers 15, 17, 19, and 21, then tests whether PANL steering propagates through those states. It uses the existing 500-case Qwen3 capture and the frozen Steering split (50 image-side plus 31 text-side tests).

The four conditions are C0 clean PANL/clean CLE, C1 steered PANL/naturally steered CLE, C2 steered PANL/clean CLE restoration, and C3 clean PANL/steered CLE transplant. PANL steering is applied at layers 14, 16, and 18 with alpha -5 and +5. Only the nine strictly downstream PANL/CLE layer pairs are run.

Probe preparation and training are CPU-only:

```bash
python "qwen3 review/SA_trajectory/PANL2CLE/run_pipeline.py" --stage prepare
OMP_NUM_THREADS=1 python "qwen3 review/SA_trajectory/PANL2CLE/run_pipeline.py" --stage train-probes
```

Run the alpha-zero GPU gate and smoke experiment:

```bash
python "qwen3 review/SA_trajectory/PANL2CLE/run_pipeline.py" --stage alpha-zero --resume
python "qwen3 review/SA_trajectory/PANL2CLE/run_pipeline.py" --stage all --smoke --resume
```

Formal execution is deliberately gated:

```bash
python "qwen3 review/SA_trajectory/PANL2CLE/run_pipeline.py" --stage run --run-formal --num-gpus 1 --resume
python "qwen3 review/SA_trajectory/PANL2CLE/run_pipeline.py" --stage analyze --resume
```

The formal physical grid is 3,483 forwards. Analysis expands shared C0/C1 trials into 5,832 four-cell rows and reports final soft-SA and CLE-probe SA using answer-equal macro, overall micro, image-side, and text-side aggregation.
