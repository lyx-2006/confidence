# Faithful Check

This directory implements the gated four-cell counterfactual experiment for Qwen3-VL.
It selects at most 70 easy and 30 hard conflict images, validates both original and
counterfactual modalities independently, recolors only the visible target fill, and
compares signed CMA with signed short-prompt Soft-SA.

Run data/model validation without loading model weights:

```bash
python "qwen3 review/short prompt check/faithful check/run_pipeline.py" --preflight-only
```

Run or resume the experiment:

```bash
python "qwen3 review/short prompt check/faithful check/run_pipeline.py" --resume
```

Run a two-case GPU smoke test in a separate output directory:

```bash
python "qwen3 review/short prompt check/faithful check/run_pipeline.py" \
  --smoke --output-root "qwen3 review/short prompt check/output/faithful_check_smoke"
```

Analyze completed trials:

```bash
python "qwen3 review/short prompt check/faithful check/analyze.py"
```

Run the reverse-prompt Soft-SA on the 110-case balanced subset and analyze it:

```bash
python "qwen3 review/short prompt check/faithful check/run_reverse_softsa.py" --resume
python "qwen3 review/short prompt check/faithful check/analyze_reverse_softsa.py"
```

The reverse scorer maps raw class 0=image and raw class 8=text back to the
canonical signed orientation where larger values mean stronger image attribution.

The default output is `qwen3 review/short prompt check/output/faithful_check/`.
JSONL files are append-and-fsync checkpoints, so `--resume` reuses every completed
single-modal gate, accepted manifest case, and completed trial. A changed model,
dataset, text pool, prompt, quota, or tolerance is rejected by the stored fingerprint.
