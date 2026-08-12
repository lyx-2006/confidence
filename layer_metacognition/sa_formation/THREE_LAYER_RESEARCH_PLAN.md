# Three-layer Source Reliance / Attribution Research Plan

## Objects

- `Actual Source Reliance` (`R_source`): behaviorally defined from fixed-answer
  evidence deletion and symmetric replacement.  No Source Attribution request is
  present when it is measured.
- `Candidate Internal Source Attribution` (`Z_attr`): a latent candidate which
  must generalize across report protocols.  It is not assumed to exist and is
  not identified with the old L18 Ridge coordinate.
- `Verbal SA Report`: the emitted 9/3/2-class report.  Protocol/token effects are
  part of this measurement, not part of behavioral ground truth.

History is a causal instrument used to separate the three objects; it is not
the final object of study.

## Experiment 1 — Clean Actual Source Reliance

Use an answer-only prompt and define the natural fixed endpoint from the Full
restricted distribution:

`A* = argmax_a logit(a | Full)`.

For the same `A*` measure:

`D = log P(A* | NoText) - log P(A* | NoImage)`

`R_j = log P(A* | ReplaceText_j) - log P(A* | ReplaceImage_j)`

Each replacement pair uses text and image from the same matched donor.  A
second deterministic donor is a reliability replicate.  Positive values are
imageward.  Verbal SA is never used to build the target.

Two estimands are retained:

- raw shared reliance, including the naturally chosen answer side;
- graded shared reliance after training-fold-only control of final side, answer
  identity, difficulty, prior strength, and Full answer margin.

The shared score is the predeclared equal-weight mean of fold-standardized
indicators; the orthogonal difference is saved as method disagreement.

Development uses the frozen existing 100-item cohort.  If its measurement gate
passes, the rule is frozen and evaluated on the remaining unique items before
the cohorts are combined for representation learning.

Measurement gate:

- at least 90 valid unique development items and zero SA leakage;
- raw deletion/replacement Spearman bootstrap lower bound above zero and sign
  agreement at least .70;
- graded residual Spearman lower bound above zero and sign agreement at least
  .60;
- at least four of five folds have positive association;
- two-indicator reliability at least .60;
- donor-1/donor-2 replacement reliability has positive lower bound and at
  least .60 standardized reliability.

## Experiment 2 — Actual Reliance Representation

Capture answer-only decoder-block outputs without an SA request at:

- `pre_answer`: the final `**Answer**:` prefix token, before `A*`;
- `post_answer`: the final token of teacher-forced `A*`.

Primary layers are fixed before analysis: 8, 12, 16, 20, 24, and 27.  All target
construction, hidden nuisance projection, scaling, alpha selection, and cell
selection occur inside item-grouped outer folds.  The internal direction gate
requires OOF R2 above zero, a positive item-bootstrap rank lower bound, at least
four positive fold effects, hidden-only R2 above a separately fitted
nuisance-only model, and bidirectional deletion-to-replacement cross-method
prediction.  Because that R2 comparison is not nested, it is not called
conditional or incremental R2.  A separate non-gating sensitivity fits nested
nuisance-only and nuisance-plus-hidden calibration models on development OOF
predictions and applies both unchanged to confirmatory items.

The same frozen fold readout is also evaluated against fresh donor-3/4
replacement targets.  This is a prospective donor-measurement sensitivity on
the same items, not a retroactive repair of the original confirmatory gate.

Only a direction passing the target and representation gates may be called a
candidate `source-use representation`.

The frozen fold directions are then applied zero-shot to the existing
Text-first/Image-first History contexts.  Population mean shift and item-level
alignment with behavioral `delta R_source` are reported separately.

## Experiment 3 — Candidate Attribution Component

The existing 80-item common-template protocol panel is first used for a zero-GPU
feasibility screen.  A single item-OOF shared direction must predict a
training-fold-derived cross-protocol semantic target.  Protocol-specific affine
calibration is allowed only for the separately labelled `rank transport`
analysis.

Coordinate invariance requires one raw/unit direction and one train-derived
origin/scale, with no protocol-specific calibration.  Its gate additionally
requires protocol agreement ICC, paired-mean/slope/intercept equivalence,
legacy-grammar holdout, covariate and item-permutation controls, and performance
above matched random directions.

If the CPU screen only supports rank transport, no invariant-state claim is
made.  Confirmatory GPU work is allowed only after the screen passes and must
add common-template random label bijections, class-row-order controls, and an
identical answer-only prefix followed by a new user SA-query branch.

The confirmatory panel uses the 76 completed Actual-Reliance confirmatory
items, which are disjoint by item from the 80-item attribution development
panel.  It freezes each behaviorally selected `A*`, the seven common protocol
specifications, every fold-specific target transform, and every L18 direction.
The joint-report branch tests new-item rank transport.  Random-bijection and
row-order prompts test protocol sensitivity.  The post-answer query branch is
reported separately because its SA-prefix hidden state is at a different
causal position and cannot be treated as the original PANL coordinate.

## Experiment 4 — Shared versus Divergent Computation

Only after both representations pass their own gates:

- compare layer timing and same-site geometry against random controls;
- run 2x2 OOF cross-decoding (`Z_R -> R`, `Z_R -> attribution`,
  `Z_attr -> attribution`, `Z_attr -> R`);
- residualize each representation against the other and nuisance covariates;
- compare Evidence and History treatment effects at population and item levels.

Direct cosine is reported only in a common layer/position basis.  Across
different positions, use cross-decoding or CKA instead.

## Experiment 5 — Causal Divergence Tracing

Blockwise patching or steering is prohibited until Experiments 1–4 validate a
behavioral target and at least one internal representation.  Causal work then
asks where a source-use state is preserved, transformed, or reconstructed into
an attribution/report state.  It must include answer-behavior and verbal-report
outcomes, bidirectional patches, same-history and different-item controls, and
held-out items.

Failure of a gate produces an explicit skipped artifact.  Natural decodability,
rank transport, or a nonzero population mean alone never authorizes mediation
or a claim of faithful instance-wise introspection.
