# Layer Metacognition Main Experiment

This package runs the Qwen2.5-VL layer-wise answer-cognition (AC), post-answer
newline (PANL), and confidence-cognition (CC) experiment. It imports the
existing `QwenVLInference` loader and does not modify the original inference,
confidence-analysis, colour-pool, or dataset files.

## CPU tests (no model-weight or GPU load)

```bash
python layer_metacognition/smoke_test_main.py \
  --cpu-only \
  --model-path /root/autodl-tmp/qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct \
  --dataset datasets/dataset_with_images.json \
  --image-dir datasets \
  --max-items 1
```

## One-case GPU smoke test

Run this after CUDA is enabled. Even though `--max-items` selects source
dataset items, the smoke test deliberately executes only the first expanded
case.

```bash
python layer_metacognition/smoke_test_main.py \
  --model-path /root/autodl-tmp/qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct \
  --dataset datasets/dataset_with_images.json \
  --image-dir datasets \
  --max-items 1
```

## Main experiment

```bash
python layer_metacognition/run_main_experiment.py \
  --model-path /root/autodl-tmp/qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct \
  --dataset datasets/dataset_with_images.json \
  --image-dir datasets \
  --layers all \
  --save-hidden-states panl \
  --output-dir layer_metacognition/output/main \
  --resume
```

`--max-items` limits original dataset items before prior/image expansion.
`--case-id` selects one stable expanded case. `--layers` accepts `all` or a
comma-separated list such as `0,4,8,12`. Results and failures are JSONL. Only
PANL vectors are saved, in batched CPU-FP16 shards.

The runner performs an inline one-case GPU preflight before it commits its
first result. AC and CC last-layer logits must match reconstructed
`LMHead(FinalNorm(block_output))` logits or the run is rejected.

## Compact result JSON

```bash
python layer_metacognition/analyze_main_results.py \
  --results layer_metacognition/output/main/results.jsonl \
  --output layer_metacognition/output/main/analysis_minimal.json \
  --summary-output layer_metacognition/output/main/summary.json
```

Each layer is stored as:

```text
[answer, answer_probability, answer_entropy_nats, soft_confidence]
```

Each layer tuple is serialized on one line. `summary.json` aggregates valid
case counts, answer distributions, mean answer probability, mean answer
entropy, and mean soft confidence by layer.

## Outputs

```text
layer_metacognition/output/main/
├── results.jsonl
├── failures.jsonl
├── metadata.json
├── analysis_minimal.json
├── hidden_states/
│   ├── index.json
│   └── shard_*.pt
└── figures/
```

Mean-embedding patching, same/cross-layer transfer, linear probes, steering,
training, and external APIs are intentionally out of scope.
