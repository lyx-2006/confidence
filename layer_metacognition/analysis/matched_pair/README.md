# V3/V4 Source Attribution matched-pair analysis

This analysis reads the completed V3/V4 experiment and its existing unimodal
model-behavior labels. It does not run the model or read/regenerate hidden
states.

Run from the repository root:

```bash
python -m layer_metacognition.analysis.matched_pair.analyze
```

Inputs:

- `layer_metacognition/output/v3_v4/results.jsonl`
- `layer_metacognition/output/v3_v4/config.json`
- `layer_metacognition/output/v3_v4/probe/text_only_labels.jsonl`
- `layer_metacognition/output/v3_v4/probe/image_only_labels.jsonl`

Outputs are written to `layer_metacognition/output/v3_v4/matched_pair/`.

Pairs match exactly on `item_id`, `prior_index`, `version`, and
`generated.current_answer`. All deltas are `condition_b - condition_a`.
Directional contrasts put the hypothesized larger condition in `condition_b`.

`SA_a` and `SA_b` are raw Semantic Patchscope `soft_image_score` values at
layer 27. `SA_late` is the arithmetic mean over layers 24–26. Answer
competition uses only existing model-behavior labels and is computed only when
`text_only_answer != image_only_answer`:

```text
M_l = log(P(image_only_answer)) - log(P(text_only_answer))
```

The conflict image-only label comes from the existing `conflict_easy`
image-only run for that item and is reused as the model-behavior image answer
when evaluating `conflict_hard`. No dataset ground-truth, `text_ans`, or
`conflict_ans` field is used.

The default underpowered rule is fewer than 20 strict pairs or fewer than 10
unique items. Underpowered contrasts retain descriptive statistics but omit
bootstrap intervals and Wilcoxon tests. Bootstrap confidence intervals resample
`item_id` clusters with replacement.
