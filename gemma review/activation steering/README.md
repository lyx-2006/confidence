# Gemma 3 delayed-SA activation steering

This experiment is a Gemma-only migration of the `dp_SA` activation-steering
experiment. It imports the prompt text from `dp_SA/prompts.py`, but contains no
Qwen runtime, image-processing, or token-alignment path.

## Small GPU smoke

The smoke run captures only layer 22 at all six positions for 30 item-distinct
samples. It then runs 10 test samples over six positions and alphas `-2, 0, 2`
(180 true-direction cells).

```bash
python "gemma review/activation steering/run_pipeline.py" --smoke
```

Resume an interrupted smoke run with `--resume`.

## Formal experiment

Formal capture stores every layer from L6 through L33. Steering layers have no
implicit default and must be selected explicitly:

```bash
python "gemma review/activation steering/run_pipeline.py" \
  --steering-layers 12 17 22 24 29 32 \
  --resume
```

By default, shuffled PANL controls run at L22/L24 when those layers are in the
selection. Override with `--shuffled-layers`; pass the flag with no values to
disable shuffled controls.

Stages can also be run separately:

```bash
python "gemma review/activation steering/capture.py" --resume
python "gemma review/activation steering/steering.py" --steering-layers 22 --resume
python "gemma review/activation steering/analysis.py"
```

Formal outputs are written to `output/results`; smoke outputs are isolated in
`output/smoke`. A changed model, dataset, prompt, capture grid, or steering grid
requires a fresh output root.

